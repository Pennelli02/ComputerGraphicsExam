import argparse
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

def split_frames(opts):
    pass


def main(opts):
    if opts.extractEN:
        extract_frames(opts)
    else:
        split_frames(opts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='YAML Configuration file')
    opts = yaml.load(open(parser.parse_args().config), Loader=yaml.Loader)
    opts = SimpleNamespace(**opts)


    main(opts)