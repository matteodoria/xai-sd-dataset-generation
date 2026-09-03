# Il percorso — come siamo arrivati ai risultati

Documento di lavoro, in italiano, scritto per ricordare **le decisioni** e non solo le
conclusioni. Il *cosa* abbiamo trovato sta in [results.md](results.md); qui
c'è il *come*, inclusi i vicoli ciechi — che sono la parte che insegna qualcosa.

---

## 1. Da dove siamo partiti

Il paper ESANN 2024 adatta Stable Diffusion a generare dataset sintetici
class-conditional. Sostituisce il Text-Encoder con un **Class-Encoder** che mappa un
vettore one-hot nello spazio di conditioning, poi ottimizza con una pipeline in
quattro passi (transfer learning, ottimizzazione bayesiana di IS e UGS, fine-tuning
del diffusion model, seconda ottimizzazione).

Il paper dimostra due cose: **che** funziona (il CAS cresce a ogni passo) e **che**
UGS ed epoca di adattamento dominano la fANOVA. Non dice **perché**. Ed è esattamente
lo spazio in cui un progetto di xAI ha senso: non aggiungere una tecnica, ma spiegare
un modello che già esiste e di cui si conoscono le prestazioni ma non i meccanismi.

La prima decisione è stata **dove guardare**. Le opzioni erano tre: lo spazio di
conditioning (cosa il modello riceve), la cross-attention (come lo usa), la dinamica
temporale (quando lo usa). Le abbiamo affrontate in quest'ordine, dalla più economica
alla più costosa — un ordine che si è rivelato fortunato, perché la prima ha
predetto il fallimento della seconda.

---

## 2. Il percorso

### Fase 1 — Lo spazio di conditioning

Leggendo `Models/class_encoder.py` è saltata all'occhio una cosa: il Class-Encoder è
`Dense(500)` seguito da `Dense(76800, use_bias=False)`, **entrambi senza attivazione**.
Nessuna non-linearità in mezzo significa che le due matrici collassano in una:

```
context(x) = (x W1 + b1) W2 = x E + u,   con  E = W1 W2,  u = b1 W2
```

Se è vero, gli embedding di classe si leggono **dai pesi**, senza generare nulla.
Ed è vero anche di più: il contesto unconditional usato dalla guidance è un input di
zeri, quindi esattamente `u`; e la direzione che UGS amplifica è `E[c]`, la riga
c-esima. Tutta l'informazione di classe passa da lì e da nient'altro.

**La decisione importante è stata verificarlo invece di darlo per buono.** Era una
deduzione dalla lettura del codice, non una misura, e su di essa poggiava tutto il
resto. Il controllo ha prodotto due sorprese.

La prima: la differenza fra modello e forma chiusa era `4.1e-4`, sopra la tolleranza
di `1e-4` che avevo fissato. Ma la tolleranza era **assoluta**, e su valori di
ampiezza ~9 non ha senso: `4.1e-4` è `4.5e-5` in termini relativi, cioè arrotondamento.
Lezione: una soglia va definita rispetto alla scala di ciò che misura.

La seconda, più interessante: confrontando entrambi con un riferimento in `float64`,
la **forma chiusa risultava cento volte più accurata del modello** (`3.9e-6` contro
`4.1e-4`). Il motivo l'ha rivelato un messaggio nei log — `TensorFloat-32 will be used
for the matrix multiplication` — TensorFlow su GPU recenti tronca la mantissa a 10 bit.
Il gate è finito dentro `scripts/xai_conditioning.py` ed è rieseguito a ogni run.

Con la forma chiusa validata, la geometria: **dal 87% al 99% dell'energia del
conditioning è comune a tutte le classi**. L'informazione discriminante vive nel poco
che resta. E i coseni fra classi grezzi sono tutti attorno a +0.96, il che rende
`cat`-`dog` simile quanto `cat`-`truck`: la struttura è nascosta dalla componente
condivisa, e appare solo dopo aver **centrato**.

Centrando, appaiono le coppie sensate — `automobile↔truck`, `cat↔dog`, `airplane↔ship`.
Qui la decisione è stata **non fermarsi all'impressione**: dieci coppie sensate su
dieci classi possono nascere dal caso, e l'occhio è generoso perché sa già quali
coppie *dovrebbero* stare insieme. Serviva un test.

Il test di permutazione usa un raggruppamento **esterno** — veicoli/animali su
CIFAR-10, le venti superclassi ufficiali su CIFAR-100 — mai scelto guardando i dati.
Su CIFAR-10, con dieci classi divise 4/6, le partizioni possibili sono solo 210: le
abbiamo enumerate **tutte**, ottenendo un p-value esatto invece di una stima. Risultato:
la divisione veicoli/animali è la **migliore fra tutte le 210**, p = 0.0048, il minimo
ottenibile. Su CIFAR-100 la separazione osservata è sei volte il massimo di diecimila
raggruppamenti casuali.

Poi **BloodMNIST ha fallito**, ed è stato l'episodio più istruttivo dell'intero
progetto. Non esistono etichette coarse ufficiali, quindi ho scelto io la divisione
granulociti / non-granulociti: è la distinzione morfologica primaria in ematologia,
è visiva prima che biologica, ed è bilanciata 4/4. Risultato: **p = 0.77**, con
separazione *negativa*.

La tentazione era provare un'altra tassonomia. Con 35 partizioni possibili, cercare
quella che funziona **garantisce di trovarla**, e il p-value che ne esce non vale
nulla. La soluzione non era un raggruppamento migliore ma **smettere di usare
raggruppamenti**: confrontare la geometria degli embedding con la similarità visiva
misurata **dai pixel delle immagini reali**, senza alcuna scelta lasciata a noi.

Il confronto fra due matrici di similarità richiede un **test di Mantel**, non una
correlazione ordinaria: le entrate non sono osservazioni indipendenti, perché ogni
classe compare in n−1 celle. Il nullo permuta l'ordine delle classi di una matrice,
preservandone la struttura e distruggendo la corrispondenza.

Esito: **BloodMNIST ha la correlazione più alta dei sei dataset (+0.878)**. Il dataset
dove la tassonomia falliva è quello che aderisce meglio all'apparenza. La ragione è
che la colorazione di Giemsa esiste *apposta* per rendere visivamente diverse cellule
biologicamente sorelle: le due referenze non solo divergono, si oppongono. Non era il
modello ad aver imparato male, ero io ad avergli chiesto la cosa sbagliata.

### Fase 2 — La cross-attention

L'idea era trasferire DAAM: le mappe di cross-attention del UNet come saliency del
generatore, da confrontare con la Grad-CAM del classificatore a valle.

Prima l'infrastruttura, che è stata la parte ingegneristicamente più delicata. I pesi
di attenzione in `keras_cv` sono una **variabile locale** dentro `CrossAttention.call`:
non esiste hook a cui agganciarsi. Abbiamo sostituito il metodo della classe con una
copia che li mette da parte, verificando che l'output restasse **bit-identico**.

Ma catturarli richiede la modalità **eager**: `predict_on_batch` avvolge il modello in
una `tf.function`, e dentro un grafo tracciato i tensori raccolti sarebbero
placeholder simbolici, non valori. Abbiamo quindi riscritto il loop di sampling
replicando `generate_image`, con un'invariante da rispettare: **le immagini generate
devono essere le stesse**. Misurato: 1 livello su 255 di differenza, su lo 0.7% dei
pixel — l'arrotondamento fra eager e graph mode.

Poi il primo ostacolo concettuale. Volevo una "mappa di salienza" mediando l'attenzione
sui token. Ma la softmax è presa **sui 100 slot**, non sulle posizioni spaziali: ogni
posizione ha una distribuzione che somma a 1, quindi mediare sui token dà una mappa
costante a 1/100 ovunque. Zero informazione. La quantità che varia spazialmente è
un'altra: quanto la distribuzione *cambia* fra passata condizionale e unconditional.

Con quella, le mappe sono risultate **quasi uniformi** (max/mean fra 1.04 e 1.16).
Stavo per concludere che non c'è localizzazione — ma quei numeri erano medie su 8
teste e 5 layer, e **mediare è il modo migliore per far sparire una struttura**: se una
testa guarda in alto a sinistra e un'altra in basso a destra, la media è piatta pur
essendo entrambe selettive.

Il controllo per singola testa ha ribaltato la conclusione: la selettività **esiste**,
con coefficienti di variazione fino a 1.28. Ma un CV alto dice solo che la mappa non è
uniforme, non che sia informativa. Le teste selettive potevano rispondere alla
posizione nella griglia — un artefatto notissimo nei transformer.

Il test di stabilità (la stessa testa su otto immagini diverse) ha mostrato **due
popolazioni**: teste con correlazione > 0.9 fra immagini, cioè posizionali, e altre
con correlazione ~0, che cambiano col contenuto. Sembrava promettente.

Poi mi sono accorto di un **errore mio nel disegno del test**: stavo misurando
l'attenzione al **primo step di denoising**, quando il latente è ancora quasi puro
rumore e nessun soggetto esiste ancora, e la confrontavo col dettaglio dell'immagine
**finale**, che nascerà trenta passi dopo. Rifatto all'ultimo step, è emersa una
correlazione col dettaglio di **−0.428**, consistente su tutte le classi.

Sembrava un risultato. L'ultimo controllo l'ha chiuso: la stessa anti-correlazione
vale per l'attenzione **unconditional** (−0.382). Il fenomeno esiste ma **non è del
condizionamento**: è una proprietà della cross-attention di questa UNet, e non
trasporta informazione di classe.

Risultato negativo, ma con una spiegazione strutturale che lo rende solido. Nella
Stable Diffusion testuale esiste un token `cat` che si lega a una regione, ed è ciò che
rende possibile DAAM. Qui il Class-Encoder emette cento slot **in blocco** da un
one-hot: non esiste uno slot "orecchie". Il condizionamento è un vettore globale, e un
vettore globale non ha nulla da localizzare. **La Fase 1 aveva già predetto questo
esito**, e non ce ne eravamo accorti.

### Fase 3 — La dinamica temporale

Chiuso l'asse spaziale, restava il *quando*. E qui la decisione chiave è stata
cambiare oggetto di misura: non più l'attenzione — un meccanismo interno — ma
`ε(c) − ε(∅)`, che **è** il segnale di classe per definizione ed è ciò che la guidance
moltiplica per UGS. Niente softmax da cui difendersi, nessun confondente.

La misura dice che il segnale è **debole quando l'immagine è rumore e cresce fino a
quintuplicare** verso la fine. Da cui la conclusione ovvia: il condizionamento conta
soprattutto alla fine. Sbagliata.

`‖Δε‖` misura l'effetto **istantaneo** a un dato passo, non l'effetto sull'immagine
finale. Una spinta minuscola al primo passo viene propagata e amplificata attraverso i
trenta successivi; una grande all'ultimo passo non ha più nulla su cui propagarsi.
Distinguere le due cose richiede un **intervento**, non un'osservazione.

L'esperimento: generare condizionando **solo dentro una finestra di passi**, dallo
stesso rumore iniziale, e misurare cosa resta. Prima con finestre disgiunte (dove
conta), poi con prefissi cumulativi (quanto basta).

Le prime misure usavano l'RMSE fra immagini, e raccontavano una storia coerente: i
primi passi recuperano il 26% della distanza in pixel, gli ultimi il 7%. Il
condizionamento conta all'inizio.

Poi è arrivato il classificatore, e la storia è cambiata. Con un ResNet20 addestrato
**solo su dati reali** come giudice indipendente, i primi cinque passi portano
l'accuratezza dal 10.0% (caso) al 15.6%: **quasi nulla**. Il picco del contributo è a
metà processo.

**L'RMSE era la metrica sbagliata.** Misura quanto cambia l'immagine, e nei primi passi
cambia moltissimo perché si decide l'impianto — composizione, sfondo, colori. Ma
quell'impianto è in larga parte *indipendente dalla classe*. Da cui l'enunciato finale:
i primi passi decidono *che immagine sarà*, quelli centrali *di cosa è immagine*, gli
ultimi rifiniscono senza aggiungere classe.

Nota personale su un errore: guardando la griglia delle immagini avevo dichiarato che
condizionare metà dei passi bastava a preservare la classe — vedevo l'auto rossa, il
camion arancione. Il classificatore ha detto 35.6% contro 64.4%. **Su dieci immagini
scelte a occhio avevo visto i casi riusciti e ignorato i falliti**, che è esattamente
l'errore che la misura serve a impedire. E le ultime quattro righe della griglia
sembrano identiche pur coprendo venti punti di accuratezza.

Infine le **ablation**, che rendono difendibile tutto il resto. La più forte non è la
randomizzazione dei pesi (9.0%, livello del caso) ma la **permutazione**: dare a ogni
classe l'embedding della successiva. Risultato: 5.6% rispetto all'etichetta richiesta —
*sotto* il livello del caso, perché quella classe non viene mai prodotta di proposito —
e **65.0% rispetto all'embedding effettivamente fornito**, contro il 64.4% del modello
intatto. Il condizionamento non fa genericamente qualcosa: **reindirizza** la
generazione con la stessa efficacia. Un artefatto del pipeline sopravvivrebbe alla
randomizzazione, non a questo.

### Fase 4 — Sei dataset, e un criterio diagnostico

Esteso l'esperimento a tutti e sei, quattro hanno replicato la campana. Due no, e sono
stati i più informativi.

**DermaMNIST**: accuratezza `all` = 14.3%, che è esattamente 1/7 — il giudice assegna
tutto a una classe sola. Prima di dire "il generatore non funziona" bisognava escludere
che fosse colpa del *nostro* codice: rigenerando con `scripts/generate_dataset.py`, lo script
originale del paper, sono usciti gli stessi artefatti fluorescenti.

E lanciando l'analisi della Fase 1 su DermaMNIST è comparso il numero che ha chiuso il
cerchio: **effective rank 1.26 su 6**. Le sette classi collassano in *una sola*
direzione. Lo spazio di conditioning non le separa, quindi il generatore non può
produrle distinte. **La Fase 1 spiega la Fase 3.**

Da lì la previsione: RetinaMNIST, che genera male (29.2%), dovrebbe avere rango basso.
Verificato: **1.29 su 4**. E i quattro che funzionano stanno tutti sopra 3.5.

Ne esce un **criterio diagnostico**: un numero calcolabile dai soli pesi dell'encoder,
in pochi secondi e senza generare un'immagine, che separa i casi in cui l'adattamento
funzionerà da quelli in cui fallirà.

---

## 3. Cosa abbiamo trovato

1. **La geometria del conditioning space riproduce la similarità visiva dei dati
   reali** (Mantel da +0.24 a +0.88, significativo su 4 dataset su 6), partendo da
   vettori one-hot ortogonali e senza mai vedere parole né categorie.
2. **La cross-attention non localizza** — risultato negativo, con tre misure
   indipendenti e una spiegazione strutturale.
3. **L'identità di classe si costruisce a metà del denoising**, mentre i pixel si
   decidono all'inizio: due fasi distinte e misurabili separatamente.
4. **L'effective rank dei residui predice se la generazione funzionerà**: sopra 3.5 sì,
   sotto 1.3 no, nessun dataset in mezzo.

Due osservazioni collaterali che valgono la pena. Su CIFAR-10 e CIFAR-100 le immagini
generate sono **classificate meglio di quelle reali** (88.4% contro 84.3%): il
generatore produce esemplari prototipici mentre il dataset reale contiene occlusioni e
casi ambigui — è il meccanismo dietro l'affermazione del paper che il sintetico a volte
batte il reale. E passare dagli iperparametri sbagliati a quelli ottimali vale **24
punti di accuratezza**, conferma indipendente del peso che la fANOVA attribuisce a IS
e UGS.

---

## 4. Quello che non ha funzionato

Cinque ipotesi scartate. Vanno riportate: sono risultati, e sono ciò che rende
credibile il resto.

**La compressione dello spazio spiega il CAS basso di CIFAR-100.** Sembrava solido:
15.35 direzioni effettive su 99 possibili, contro 5.33 su 9 di CIFAR-10. Il controllo
— sottocampionare CIFAR-100 a dieci classi da dieci superclassi diverse — ha dato
5.90 ± 0.56, cioè *più* di CIFAR-10. Era un artefatto della numerosità: l'effective
rank cresce sublinearmente, e normalizzarlo per il massimo teorico è ingannevole.
Ho proposto io quel controllo proprio perché poteva smentirmi, e mi ha smentito.

**La quantità di segnale discriminante predice la qualità.** No: BloodMNIST ha il
residuo più piccolo (1.2%) e genera bene, DermaMNIST il più grande (13.0%) e fallisce.
Conta come il segnale è *distribuito*, non quanto ce n'è.

**La cross-attention è una saliency del generatore.** No, per tre misure indipendenti.

**Si può risparmiare il 25% del costo della guidance.** L'idea era: se il
condizionamento serve solo nella prima metà, negli ultimi passi basta un forward pass
invece di due. La curva dell'accuratezza però non satura — cresce quasi linearmente
fino alla fine — quindi non c'è punto di taglio da sfruttare.

**Il contributo è massimo nei primi passi.** Dedotto dall'RMSE, smentito
dall'accuratezza. Il picco è al centro.

---

## 5. Le lezioni di metodo

Questa è la parte da rileggere.

**Verifica lo strumento prima di usarlo.** Il gate sulla linearità sembrava una
formalità e ha scoperto il TF32. Ogni run lo riesegue e si ferma se fallisce: un
controllo che gira una volta sola non protegge da nulla.

**Una soglia va rapportata alla scala.** La tolleranza assoluta di `1e-4` avrebbe
bocciato un risultato corretto. E più tardi, su RetinaMNIST, il criterio sul bias ha
fallito perché rapportava il bias a un rumore che era esso stesso di pochi ULP di
`float32`.

**Le medie distruggono la struttura.** L'attenzione sembrava uniforme perché mediata
su otto teste e cinque layer. Prima di concludere che una struttura non c'è, va
cercata dove non è stata mediata via.

**Un test buono con il riferimento sbagliato fallisce.** BloodMNIST. E la reazione
giusta non è cambiare riferimento finché uno funziona — con 35 partizioni la trovi
sempre — ma sostituire la scelta con una misura.

**Misurare l'intensità di un meccanismo non dice nulla sulla sua importanza causale.**
`‖Δε‖` è massimo dove il condizionamento conta di meno. Per l'importanza serve
intervenire.

**La metrica sbagliata racconta una storia coerente e falsa.** L'RMSE dava una
narrazione plausibile e completamente diversa da quella vera. Quando due metriche
divergono, la domanda non è quale credere ma *quale risponde alla domanda posta*.

**Guardare le immagini non basta.** Ci sono cascato dichiarando che metà dei passi
bastava. Su dieci esempi si vedono i successi e non i fallimenti.

**Il controllo che può smentirti va fatto per primo.** Il sottocampionamento di
CIFAR-100 e la permutazione degli embedding sono i due momenti in cui il progetto ha
guadagnato più credibilità — uno smentendo un'ipotesi, l'altro confermandola in modo
che un artefatto non avrebbe potuto superare.

**Un'ipotesi esclusa vale quanto una confermata.** Cinque su cinque, in questo
progetto, hanno indirizzato dove guardare dopo.

**I risultati negativi vanno in evidenza, non in appendice.** La cross-attention che
non localizza, con la ragione strutturale, dice più di una mappa colorata che convince
senza dimostrare.

---

## Dettagli tecnici da ricordare

- **TF32**: TensorFlow su GPU Ampere+ esegue i matmul `float32` con mantissa a 10 bit.
- **Il numero di passi eseguiti non è quello richiesto**:
  `tf.range(1, 1000, 1000 // num_steps)` dà 32 valori per 31, 45 per 44, 50 per 48.
  Vale anche per gli iperparametri ottimali del paper.
- **PathMNIST ha domain shift**: il `Best Score` di 98.9% è *validation*; sul test set,
  raccolto in un centro clinico diverso, il giudice fa 88.5%.
- **Il `ClassEncoder` ingoia gli errori di caricamento** e prosegue con pesi casuali.
  Un path sbagliato non dà un crash ma un'analisi di puro rumore dall'aria plausibile.
- **Il codice istanzia la UNet di Stable Diffusion 1.x** (contesto 768, 8 teste, 16
  blocchi di attenzione), mentre il paper dichiara SD 2.0. Da chiarire con Lomurno.

# HTML generato da docs/render.py a partire da percorso.md (serve solo per
# la stampa in PDF): si rigenera con `python docs/render.py docs/percorso.md`.
docs/percorso.html