"""
Etapa 2: Descripcion y busqueda sobre recortes ya extraidos (CLI)
-----------------------------------------------------------------
Carga FastVLM UNA sola vez en este proceso (antes: un subprocess por recorte
que recargaba el modelo). La logica vive en pipeline/describe.py y
pipeline/runner.py.

Uso:
    python search_fastvlm.py Video_51 --describir
    python search_fastvlm.py Video_51 --consulta "playera roja, pantalon de mezclilla" --modo clip
    modos: clip | hibrido | texto | lexico
"""

import argparse

from pipeline.runner import MODOS, Motor
from pipeline.video_io import fmt_tiempo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="nombre del video en salidas/")
    ap.add_argument("--consulta")
    ap.add_argument("--modo", choices=MODOS, default="clip")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--describir", action="store_true", help="describe el mejor recorte de cada persona")
    ap.add_argument("--dispositivo", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    motor = Motor(overrides={"dispositivo": args.dispositivo} if args.dispositivo else None)
    progreso = lambda f, msg: print(f"  {msg}", end="\r")
    if args.describir:
        descs = motor.describir(args.video, None, progreso)
        print(f"\n{len(descs)} descripciones -> salidas/{args.video}/descripciones.json")
        print(motor.stats(args.video).get("descripcion"))
    if args.consulta:
        for r in motor.buscar(args.video, args.consulta, args.modo, args.top_k, progreso):
            print(f"#{r['posicion']:>2} track {r['track_id']:>5} | t={fmt_tiempo(r['timestamp_sec'])} "
                  f"| score {r['score']:.3f} | {r['descripcion'][:90]}")


if __name__ == "__main__":
    main()
