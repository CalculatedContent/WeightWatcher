import pandas as pd

from . import remove_traps as remove_traps_ops
from . import weightwatcher as wwcore


def analyze_traps(
    watcher,
    model=None,
    layers=None,
    min_evals=wwcore.DEFAULT_MIN_EVALS,
    max_evals=wwcore.DEFAULT_MAX_EVALS,
    min_size=None,
    max_size=None,
    max_N=wwcore.DEFAULT_MAX_N,
    glorot_fix=False,
    plot=False,
    savefig=wwcore.DEF_SAVE_DIR,
    conv2d_norm=True,
    ww2x=wwcore.DEFAULT_WW2X,
    pool=wwcore.DEFAULT_POOL,
    conv2d_fft=False,
    fft=False,
    channels=None,
    svd_method=wwcore.FAST_SVD,
    start_ids=wwcore.DEFAULT_START_ID,
    base_model=None,
    peft=wwcore.DEFAULT_PEFT,
    rng=None,
):
    """Externalized implementation for WeightWatcher.analyze_traps()."""
    if layers is None:
        layers = []

    watcher.set_model_(model, base_model)

    if min_size or max_size:
        wwcore.logger.warning("min_size and max_size options changed to min_evals, max_evals, ignored for now")

    if ww2x:
        wwcore.logger.warning("WW2X option deprecated, reverting too POOL=False")
        ww2x = False
        pool = False

    params = wwcore.DEFAULT_PARAMS.copy()
    params[wwcore.MIN_EVALS] = min_evals
    params[wwcore.MAX_EVALS] = max_evals
    params[wwcore.MAX_N] = max_N

    params[wwcore.PLOT] = plot
    params[wwcore.RANDOMIZE] = True
    params[wwcore.MP_FIT] = True
    params[wwcore.GLOROT_FIT] = glorot_fix
    params[wwcore.CONV2D_NORM] = conv2d_norm

    params[wwcore.POOL] = pool
    params[wwcore.WW2X] = ww2x
    params[wwcore.CONV2D_FFT] = conv2d_fft
    params[wwcore.FFT] = fft

    params[wwcore.CHANNELS_STR] = channels
    params[wwcore.LAYERS] = layers
    params[wwcore.STACKED] = False

    params[wwcore.DETX] = False
    params[wwcore.SVD_METHOD] = svd_method
    params[wwcore.TOLERANCE] = wwcore.WEAK_RANK_LOSS_TOLERANCE
    params[wwcore.START_IDS] = start_ids

    params[wwcore.SAVEFIG] = savefig
    params[wwcore.PEFT] = peft
    params[wwcore.INVERSE] = False
    params["rng"] = remove_traps_ops._normalize_trap_rng(rng=rng)

    wwcore.logger.debug("params {}".format(params))
    if not watcher.valid_params(params):
        msg = "Error, params not valid: \n {}".format(params)
        wwcore.logger.error(msg)
        raise Exception(msg)
    params = watcher.normalize_params(params)

    layer_iterator = watcher.make_layer_iterator(model=watcher.model, layers=layers, params=params, base_model=watcher.base_model)
    trap_rows = []

    for ww_layer in layer_iterator:
        if not ww_layer.skipped and ww_layer.has_weights:
            watcher.apply_normalize_Wmats(ww_layer, params)

            if params[wwcore.FFT]:
                watcher.apply_FFT(ww_layer, params)

            layer_rows = watcher.apply_analyze_traps(ww_layer, params=params)
            if layer_rows:
                trap_rows.extend(layer_rows)

    if len(trap_rows) > 0:
        details = pd.DataFrame.from_records(trap_rows)
    else:
        details = pd.DataFrame(columns=watcher._trap_result_columns())

    trap_cols = watcher._trap_result_columns()
    details = details.reindex(columns=trap_cols + [c for c in details.columns if c not in trap_cols])

    if len(details) > 0:
        lead_cols = ["layer_id", "name"]
        details = details[lead_cols + [c for c in details.columns if c not in lead_cols]]

    watcher.details = details
    return details
