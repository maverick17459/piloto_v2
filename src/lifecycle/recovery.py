# src/lifecycle/recovery.py
def recover_stale_runs(plan_run_store, store, log):
    """
    Si el server se recarga (uvicorn --reload), cualquier run queued/running
    queda colgado en memoria. Lo marcamos como error para que el UI no se quede
    "ejecutando" eternamente.
    """
    try:
        runs = plan_run_store.list_all()
    except Exception as e:
        log.info(f"event=recover.list_all.error err={type(e).__name__}:{e}")
        return

    recovered = 0
    for r in runs:
        if getattr(r, "status", None) in ("queued", "running"):
            recovered += 1
            plan_run_store.update(
                r.run_id,
                status="error",
                last_event="recovered_after_reload",
                error="Run detenido por recarga del servidor (uvicorn --reload).",
            )
            try:
                store.add_message(
                    r.chat_id,
                    "assistant",
                    f"⚠️ El plan (run_id={r.run_id}) fue detenido por recarga del servidor. "
                    "Vuelve a confirmarlo si quieres ejecutarlo.",
                )
            except Exception:
                pass

    log.info(f"event=recover.done recovered={recovered}")
