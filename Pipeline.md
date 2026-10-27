# Pipeline
In questo file presentiamo la pipeline del progetto che si pone di risolvere il nostro obiettivo.

---
## Obiettivo

Realizzare una pipeline che ricostruisca una scena statica a partire da un video e studiare come il numero e la selezione dei frame influenzino:

- qualità delle viste generate
- tempo di addestramento
- memoria occupata
- numero finale di Gaussiane

**La domanda di ricerca:** possibile ridurre il numero di immagini utilizzate da 3D Gaussian Splatting mantenendo una qualità visiva simile e diminuendo il costo computazionale?

---

## Struttura Progetto

### VideoDirectory
In questa cartella verrà messo il video dell'oggetto da ricostruire (truck.mp4 e miovideo)

### FrameSelection.py
script di python che usa FFMPEG e pyAV per gestire il video e selezionare i frame che ci interessano in base al config file e fa anche lo splitting tra training e testing

### config.yaml
file dove saranno presenti i parametri di configurazione del progetto ad eccezione della gestione del training delle gaussiane 

### Gaussian Splatting
gestito su colab dove un notebook prende i risultati ottenuti da colmap e ci fornisce i risultati (file.ply e foto finali per metriche)

### Colmap 
gestito esternamente attraverso la GUI. Noi forniamo gli input e gli prendiamo l'output

### Testing.py
script per il testing e metriche

### Visualizzazione
useremo supersplat per la visualizzazione