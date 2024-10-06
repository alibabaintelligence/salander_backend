from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta
import io
import os
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_local_maxima(data):
    return np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1

def select_top_maxima(times, values, n_maxima, time_range):
    sorted_indices = np.argsort(values)[::-1]
    selected_indices = []
    for idx in sorted_indices:
        if len(selected_indices) == n_maxima:
            break
        if not selected_indices or all(abs(times[idx] - times[i]) > time_range for i in selected_indices):
            selected_indices.append(idx)
    return np.sort(selected_indices)

def get_data_regions(times, selected_maxima, cft, window_before=300, window_after=600):
    cft_values = cft[selected_maxima]
    cft_max = np.max(np.abs(cft_values))
    
    regions = []
    for idx, value in zip(selected_maxima, cft_values):
        start = max(0, times[idx] - window_before)
        end = times[idx] + window_after
        regions.append((start, end, value))
    
    merged_regions = []
    for region in sorted(regions):
        if not merged_regions or merged_regions[-1][1] < region[0]:
            merged_regions.append(region)
        else:
            merged_regions[-1] = (merged_regions[-1][0], max(merged_regions[-1][1], region[1]),
                                  merged_regions[-1][2] + region[2])
        
    return merged_regions, cft_max

@app.post("/sta-lta-maxima/")
async def process_file(file: UploadFile = File(...)):
    contents = await file.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mseed') as temp_file:
        temp_file.write(contents)
        temp_file_path = temp_file.name

    try:
        stream = read(temp_file_path)

        minfreq, maxfreq = 0.5, 1.0
        stream_filt = stream.copy()
        stream_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
        trace_filt = stream_filt.traces[0].copy()
        trace_times_filt = trace_filt.times()
        trace_data_filt = trace_filt.data

        df = trace_filt.stats.sampling_rate
        sta_len, lta_len = 100, 1500
        cft = classic_sta_lta(trace_data_filt, int(sta_len * df), int(lta_len * df))

        local_maxima = find_local_maxima(cft)

        n_maxima = 10
        time_range = 200
        selected_maxima = select_top_maxima(trace_times_filt[local_maxima], cft[local_maxima], n_maxima, time_range)

        merged_rectangles, cft_max = get_data_regions(trace_times_filt, local_maxima[selected_maxima], cft)

        # Prepare data for CSV export
        df1 = pd.DataFrame({'CFT': cft, 'Time': trace_times_filt, 'Velocity': trace_data_filt})
        df2 = pd.DataFrame({'Start Time': [rect[0] for rect in merged_rectangles], 
                            'End time': [rect[1] for rect in merged_rectangles], 
                            'Percentages': [rect[2] for rect in merged_rectangles]})

        # Create in-memory CSV files
        csv_buffer1 = io.StringIO()
        csv_buffer2 = io.StringIO()
        df1.to_csv(csv_buffer1, index=False)
        df2.to_csv(csv_buffer2, index=False)

        return {
            "filename": file.filename,
            "cft_max": cft_max,
            "data_csv": csv_buffer1.getvalue(),
            "indexes_csv": csv_buffer2.getvalue()
        }

    finally:
        os.unlink(temp_file_path)

@app.get("/")
async def read_root():
    return {"message": "This API works"}