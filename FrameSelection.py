import argparse
import sqlite3
import struct

import numpy as np
import yaml
import cv2 as cv
import os
from types import SimpleNamespace


def extract_frames(opts):
    if opts.ExVideo:
        video_path = "VideoDirectory/Truck.mp4"
    else:
        video_path = os.path.join("VideoDirectory", opts.nameVid)

    # Apri il video con OpenCV
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Errore nell'apertura del video.")
        return

    dir_path = "images"

    # Crea la cartella di destinazione se non esiste
    os.makedirs(dir_path, exist_ok=True)
    if len(os.listdir(dir_path)) > 0:
        print(f"La cartella {dir_path} non è vuota: svuotala prima di estrarre di nuovo.")
        return

    print("OpenCV:", cv.__version__)




    # Leggi il numero di FPS del video
    fps = cap.get(cv.CAP_PROP_FPS)
    # Calcola dopo quanti frame estrarre (arrotondato all'intero più vicino)
    # Usiamo max(1, ...) per evitare che l'intervallo sia 0 se target_fps è maggiore di fps
    frame_interval = max(1, int(round(fps / opts.target_fps)))

    print(f"FPS del video: {fps:.2f} | Salvataggio ogni {frame_interval} frame")

    total_frames_read = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Salva il frame solo all'inizio di ogni secondo
        if total_frames_read % frame_interval == 0:
            frame_name = os.path.join(dir_path, f"frame_{saved_count:04d}.jpg")
            cv.imwrite(frame_name, frame)
            saved_count += 1

        total_frames_read += 1

    # Rilascia le risorse
    cap.release()
    print(f"Estrazione completata. Frame letti: {total_frames_read} | Totale frame salvati: {saved_count}")
#----------------------------------------------------------------------------------------------------------------
#Funzioni Helper

def get_names(opts, dir_path):
    # Tutti i frame in images/, in ordine alfabetico (= ordine temporale)
    names = sorted(f for f in os.listdir(dir_path) if f.endswith(".jpg"))
    print(f"Frame trovati: {len(names)}")
    return names

def check_colmap(opts, names):
    # images.bin comincia con un numero: quante immagini ha registrato COLMAP
    if not os.path.exists(opts.colmap_images_bin):
        print("ATTENZIONE: images.bin non trovato, non posso controllare i frame")
        return True                                            # non posso controllare, continuo

    with open(opts.colmap_images_bin, "rb") as f:
        registered = struct.unpack("<Q", f.read(8))[0]         # primi 8 byte = numero di immagini

    if registered == len(names):
        print(f"OK: COLMAP ha registrato tutti i {registered} frame")
        return True

    print(f"ERRORE: in images/ ci sono {len(names)} frame ma COLMAP ne ha registrati {registered}.")
    print("Togli da images/ i frame non registrati oppure rifai la ricostruzione.")
    return False

def load_small(path, width):
    # Legge l'immagine in bianco e nero e la rimpicciolisce (i calcoli sono più veloci)
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Impossibile leggere {path}")
    scale = width / img.shape[1]
    return cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

def compute_sharpness(names, dir_path):
    print("Calcolo nitidezza...")
    values = []                                        # lista vuota dove metto un valore per frame
    for name in names:                                 # scorro i frame in ordine (frame_0000, frame_0001, ...)
        img = load_small(os.path.join(dir_path, name), 800)   # apro images/frame_xxxx.jpg in grigio, largo 800 px (da 4K, così è più veloce)
        values.append(cv.Laplacian(img, cv.CV_64F).var())     # Laplaciano -> variazione -> aggiungo alla lista
    return np.array(values)                            # lo trasformo in array numpy

def find_blurry(sharp, ratio, window):
    blurry = np.zeros(len(sharp), dtype=bool)          # array di 250 "False": all'inizio nessun frame è mosso
    for i in range(len(sharp)):                        # per ogni frame...
        neighbors = sharp[max(0, i - window): i + window + 1]   # ...prendo la nitidezza dei vicini (da i-window a i+window)
        if sharp[i] < ratio * np.median(neighbors):    # se è molto meno nitido della mediana dei vicini
            blurry[i] = True                           # lo segno come mosso
    return blurry                                      # array di True/False

# aiutato da Claude
def compute_motion_db(names, db_path):
    # changes[i] = quanto cambia il frame i rispetto al frame i-1, usando i match già calcolati da COLMAP
    print("Leggo i match consecutivi dal database di COLMAP...")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database SQLite COLMAP non trovato in: {db_path}")
    con = sqlite3.connect(db_path)

    # id che COLMAP ha dato a ogni frame (dipende dal nome del file)
    ids = {}
    for image_id, name in con.execute("SELECT image_id, name FROM images"):
        ids[os.path.basename(name)] = image_id

    M = 2147483647                       # costante che COLMAP usa per codificare una coppia di immagini in un solo numero
    matches = []
    for i in range(len(names) - 1):
        a, b = ids[names[i]], ids[names[i + 1]]
        pair_id = min(a, b) * M + max(a, b)
        row = con.execute("SELECT rows FROM two_view_geometries WHERE pair_id = ?", (pair_id,)).fetchone()
        matches.append(row[0] if row else 0)     # nessuna riga = nessun match
    con.close()

    matches = np.array(matches, dtype=float)
    change = 1.0 - (matches / matches.max())       # tanti match = poco cambiamento (0), pochi match = molto (vicino a 1)
    return np.concatenate([[0.0], change])       # il primo frame non ha un precedente

def compute_motion_by_diff(names, dir_path, method):
    print(f"Calcolo movimento ... (metodo: {method})")
    prev = load_small(os.path.join(dir_path, names[0]), 640)  # primo frame in grigio, largo 640 px
    changes = [0.0]
    for name in names[1:]:  # dal secondo frame in poi
        cur = load_small(os.path.join(dir_path, name), 640)  # frame corrente
        a = cv.GaussianBlur(prev, (5, 5), 0).astype(float)  # sfoco leggermente il frame prima (toglie il rumore) e lo converto in decimali
        b = cv.GaussianBlur(cur, (5, 5), 0).astype(float)  # stessa cosa col frame corrente
        change = float(np.abs(a - b).mean() / 255.0)  # differenza assoluta pixel per pixel, media, divisa per 255 -> tra 0 e 1

        changes.append(change)  # salvo il cambiamento di questo frame
        prev = cur  # il corrente diventa il precedente
    return np.array(changes)

def pick_equispaced(values, n):
    targets = np.linspace(values[0], values[-1], n)    # n valori equidistanti tra il primo e l'ultimo della curva
    chosen = []                                        # frame scelti (posizioni nell'array values)
    for t in targets:                                  # per ogni valore obiettivo...
        order = np.argsort(np.abs(values - t))         # ...ordino i frame dal più vicino al più lontano da t
        for k in order:
            if int(k) not in chosen:                   # prendo il primo non ancora scelto
                chosen.append(int(k))
                break                                  # passo al prossimo obiettivo
    return sorted(chosen)                              # li rimetto in ordine temporale


def save_list(path, names):
    with open(path, "w") as f:          # apro il file in scrittura (lo crea o lo sovrascrive); "with" lo chiude da solo
        f.write("\n".join(names))       # unisco i nomi mettendo un "a capo" tra l'uno e l'altro -> un nome per riga


def create_scaled_images(factor, overwrite: bool = False):
    """Crea una directory images_{factor} con le immagini ridotte di `factor` volte (PNG)."""
    if factor < 2:
        raise ValueError("factor deve essere >= 2 (con factor=1 si usa direttamente 'images').")
    src = os.path.join("images")
    dst = os.path.join(f"images_{factor}")
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Cartella sorgente non trovata: {src}")

    os.makedirs(dst, exist_ok=True)
    names = sorted(f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        raise RuntimeError(f"Nessuna immagine in {src}")

    done = 0
    for name in names:
        out_path = os.path.join(dst, os.path.splitext(name)[0] + ".png")
        if os.path.isfile(out_path) and not overwrite:
            continue
        img = cv.imread(os.path.join(src, name))
        if img is None:
            raise RuntimeError(f"Impossibile leggere {name}")
        h, w = img.shape[:2]
        small = cv.resize(img, (round(w / factor), round(h / factor)), interpolation=cv.INTER_AREA)
        cv.imwrite(out_path, small)
        done += 1

    print(f"{dst}: {done} immagini create, {len(names)} totali in images/")

#--------------------------------------------------------------
# funzione split e frame selection

def split_frames(opts):
    dir_path = "images"                       # cartella dei frame
    split_dir = "splits"                      # cartella dove scrivo gli elenchi
    os.makedirs(split_dir, exist_ok=True)     # la creo se non esiste

    names = get_names(opts, dir_path) # prendo il nome dalla cartella colmap però non so se ha senso controllare
    if not check_colmap(opts, names):
        return  # mi fermo: le liste sarebbero sbagliate

    test_idx = list(range(0, len(names), opts.test_every))   # posizioni 0, 7, 14, ... (con test_every = 7)
    candidates = []                                          # posizioni che possono andare nel training
    for i in range(len(names)):                              # per ogni frame...
        near_test = False                                    # ...parto supponendo che non sia vicino a un test
        for t in test_idx:                                   # controllo tutti i frame di test
            if abs(i - t) <= opts.guard:                     # distanza da un test <= guard: è un test o troppo vicino
                near_test = True
        if not near_test:
            candidates.append(i)                             # altrimenti è un candidato per il training
    candidates = np.array(candidates)                        # array numpy, serve per le operazioni dopo

    save_list(os.path.join(split_dir, "test.txt"), [names[i] for i in test_idx])   # scrivo i nomi dei test in splits/test.txt
    print(f"Test: ... | Candidati per il training: ...")

    # --- 2. SEGNALI: nitidezza e movimento di ogni frame
    sharp = compute_sharpness(names, dir_path)
    blurry = find_blurry(sharp, opts.blur_ratio, opts.blur_window)
    print(f"Frame mossi: {int(blurry.sum())} su {len(names)}")

    cum = None
    if any("motion" in strategy for strategy, n in opts.experiments):
        if opts.motion_db:
            changes = compute_motion_db(names, opts.db_path)
        else:
            changes = compute_motion_by_diff(names, dir_path, "differenza immagini")
        cum = np.cumsum(changes)  # movimento cumulato: sale veloce dove la scena cambia molto

    summary = []                                       # qui raccolgo le statistiche per il riepilogo
    for strategy, n in opts.experiments:               # es. ("uniform", 50), ("baseline", None)
        for part in strategy.split("+"):               # "motion+quality" diventa ["motion", "quality"]
            if part not in ["baseline", "uniform", "motion", "quality"]:
                print(f"Strategia sconosciuta: {strategy}")
                return                                 # errore di battitura nel config: mi fermo

                # Senza N hanno senso solo "baseline" e "quality" (tutti i frame meno i mossi)
        if n is None and strategy not in ["baseline", "quality"]:
            print(f"[{strategy}] senza N equivale alla baseline: salto")
            continue

        pos = candidates  # parto da tutti i candidati
        if "quality" in strategy:
            pos = pos[~blurry[pos]]  # tolgo quelli segnati come mossi

        if strategy == "baseline":
            chosen = pos                               # baseline: tutti i frame utili
            tag = "baseline"
        elif n is None:  # qui ci arriva solo "quality"
            chosen = pos  # tutti i candidati meno i mossi
            tag = "quality_all"
        else:
            if n > len(pos):                           # N più grande dei frame disponibili: impossibile
                print(f"[{strategy}] servono {n} frame ma ce ne sono {len(pos)}: salto")
                continue                               # passo alla configurazione successiva
            if "motion" in strategy:
                values = cum[pos]                      # curva = movimento cumulato dei candidati
            else:
                values = pos.astype(float)             # curva = posizione nel tempo
            chosen = pos[pick_equispaced(values, n)]   # scelgo n frame equidistanti sulla curva
            tag = f"{strategy}_N{n}".replace("+", "_") # nome del file, es. "motion_N50"

        save_list(os.path.join(split_dir, f"train_{tag}.txt"),
                      [names[i] for i in chosen])  # scrivo i nomi in splits/train_<tag>.txt

        gaps = np.diff(chosen)  # differenze tra posizioni consecutive = distanze tra frame scelti
        summary.append((tag, len(chosen), gaps.min(), gaps.mean(), gaps.max(),
                            sharp[chosen].mean(), int(blurry[chosen].sum())))

    # --- 4. RIEPILOGO
    print("\nconfigurazione      frame   distanza min/media/max   nitidezza   mossi")
    for tag, count, gmin, gmean, gmax, sh, nblur in summary:
        print(f"{tag:<20}{count:>5}   {gmin:>6}/{gmean:>5.1f}/{gmax:>4}   {sh:>10.1f}   {nblur:>5}")


def main(opts):
    if opts.extractEN:
        extract_frames(opts)
        if opts.enable_scaling:
            create_scaled_images(opts.factor, opts.overwrite)
    else:
        split_frames(opts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='YAML Configuration file')
    opts = yaml.load(open(parser.parse_args().config), Loader=yaml.Loader)
    opts = SimpleNamespace(**opts)


    main(opts)