import numpy as np
import polars as pl
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.metrics import mean_squared_error

def lpf(wave, fs=12*60*24, fe=60, n=3):
    nyq = fs / 2.0
    b, a = butter(1, fe/nyq, btype='low')
    for i in range(0, n):
        wave = filtfilt(b, a, wave)
    return wave

def post_process_for_seg(
    keys: list[str], preds: np.ndarray, score_th: float = 0.01, distance: int = 5000
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pl.DataFrame: submission dataframe
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)

        for i, event_name in enumerate(["onset", "wakeup"]):
            before_RMSE = np.sqrt(mean_squared_error(this_event_preds, np.zeros_like(this_event_preds)))
            this_event_preds = this_series_preds[:, i]
            this_event_preds = lpf(this_event_preds)
            after_RMSE = np.sqrt(mean_squared_error(this_event_preds, np.zeros_like(this_event_preds)))
            decay_ratio = before_RMSE/after_RMSE
            print(f"decay_ratio: {decay_ratio}")
            this_event_preds *= decay_ratio
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]

            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df
