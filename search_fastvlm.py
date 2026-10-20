"""
Etapa 2: Descripcion y busqueda sobre recortes ya extraidos (CLI)
-----------------------------------------------------------------
Carga FastVLM UNA sola vez en este proceso (antes: un subprocess por recorte
que recargaba el modelo). Describe a cada persona y busca sobre ese texto; no
interviene ningun otro modelo de vision. La logica vive en pipeline/describe.py
y pipeline/runner.py.

Uso:
    python search_fastvlm.py Video_51 --describir
    python search_fastvlm.py Video_51 --consulta "playera roja, pantalon de mezclilla"
    modos: combinado (por defecto) | texto (semantico) | lexico (linea base TT1)
"""

import argparse

from pipeline.runner import MODOS, Motor
from pipeline.video_io import fmt_tiempo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="nombre del video en salidas/")
    ap.add_argument("--consulta")
    ap.add_argument("--modo", choices=MODOS, default="combinado")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--describir", action="store_true", help="describe el mejor recorte de cada persona")
    ap.add_argument("--dispositivo", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    motor = Motor(overrides={"dispositivo": args.dispositivo} if args.dispositivo else None)
    # ljust para que la linea de progreso no deje restos de la anterior
    progreso = lambda frac, msg: print(f"  {msg}".ljust(78), end="\r")
    if args.describir:
        descs = motor.describir(args.video, None, progreso)
        print(f"\n{len(descs)} descripciones -> salidas/{args.video}/descripciones.json")
        print(motor.stats(args.video).get("descripcion"))
    if args.consulta:
        resultados = motor.buscar(args.video, args.consulta, args.modo, args.top_k, progreso)
        print(f"\nModo {args.modo}: {len(resultados)} coincidencia(s)")
        for r in resultados:
            print(f"#{r['posicion']:>2} track {r['track_id']:>5} | t={fmt_tiempo(r['timestamp_sec'])} "
                  f"| score {r['score']:.3f} | {r['descripcion'][:90]}")


if __name__ == "__main__":
    main()
