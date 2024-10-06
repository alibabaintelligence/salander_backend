from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta
import io
import os
import tempfile
import traceback
import time
import psutil
import sys
import json
from datetime import datetime

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

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

@app.post("/sta-lta/moon")
async def process_file_moon(file: UploadFile = File(...)):
    metrics = {
        "start_time": time.time(),
        "file_size": 0,
        "peak_memory": 0,
        "read_time": 0,
        "filter_time": 0,
        "sta_lta_time": 0,
        "analysis_time": 0,
        "csv_time": 0,
    }

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    file_read_start = time.time()
    contents = await file.read()
    metrics["read_time"] = time.time() - file_read_start
    metrics["file_size"] = sys.getsizeof(contents) / 1024 / 1024  # in MB
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mseed') as temp_file:
        temp_file.write(contents)
        temp_file_path = temp_file.name

    try:
        stream = read(temp_file_path)
        if not stream:
            raise ValueError("Unable to read the file as a seismic stream")

        filter_start = time.time()
        minfreq, maxfreq = 0.5, 1.0
        stream_filt = stream.copy()
        stream_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
        metrics["filter_time"] = time.time() - filter_start
        
        if not stream_filt.traces:
            raise ValueError("No traces found in the filtered stream")

        trace_filt = stream_filt.traces[0].copy()
        trace_times_filt = trace_filt.times()
        trace_data_filt = trace_filt.data

        sta_lta_start = time.time()
        df = trace_filt.stats.sampling_rate
        sta_len, lta_len = 100, 1500
        cft = classic_sta_lta(trace_data_filt, int(sta_len * df), int(lta_len * df))
        metrics["sta_lta_time"] = time.time() - sta_lta_start

        analysis_start = time.time()
        local_maxima = find_local_maxima(cft)
        if len(local_maxima) == 0:
            raise ValueError("No local maxima found in the CFT")

        n_maxima = 10
        time_range = 200
        selected_maxima = select_top_maxima(trace_times_filt[local_maxima], cft[local_maxima], n_maxima, time_range)
        merged_rectangles, cft_max = get_data_regions(trace_times_filt, local_maxima[selected_maxima], cft)
        metrics["analysis_time"] = time.time() - analysis_start

        csv_start = time.time()
        df1 = pd.DataFrame({'CFT': cft, 'Time': trace_times_filt, 'Velocity': trace_data_filt})
        df2 = pd.DataFrame({'Start Time': [rect[0] for rect in merged_rectangles], 
                            'End time': [rect[1] for rect in merged_rectangles], 
                            'Percentages': [rect[2] for rect in merged_rectangles]})

        csv_buffer1 = io.StringIO()
        csv_buffer2 = io.StringIO()
        df1.to_csv(csv_buffer1, index=False)
        df2.to_csv(csv_buffer2, index=False)
        metrics["csv_time"] = time.time() - csv_start

        metrics["peak_memory"] = get_memory_usage()
        metrics["total_time"] = time.time() - metrics["start_time"]
        metrics["num_local_maxima"] = len(local_maxima)
        metrics["num_merged_regions"] = len(merged_rectangles)

        # Prepare the response JSON
        response_data = {
            "filename": file.filename,
            "cft_max": cft_max,
            "metrics": metrics,
            "data_csv": csv_buffer1.getvalue(),
            "indexes_csv": csv_buffer2.getvalue()
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        os.unlink(temp_file_path)
        

@app.post("/sta-lta/mars")
async def process_file_mars(file: UploadFile = File(...)):
    metrics = {
        "start_time": time.time(),
        "file_size": 0,
        "peak_memory": 0,
        "read_time": 0,
        "filter_time": 0,
        "sta_lta_time": 0,
        "analysis_time": 0,
        "csv_time": 0,
    }

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    file_read_start = time.time()
    contents = await file.read()
    metrics["read_time"] = time.time() - file_read_start
    metrics["file_size"] = sys.getsizeof(contents) / 1024 / 1024  # in MB
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mseed') as temp_file:
        temp_file.write(contents)
        temp_file_path = temp_file.name

    try:
        stream = read(temp_file_path)
        if not stream:
            raise ValueError("Unable to read the file as a seismic stream")

        filter_start = time.time()
        minfreq, maxfreq = 0.5, 1.0
        stream_filt = stream.copy()
        stream_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
        metrics["filter_time"] = time.time() - filter_start
        
        if not stream_filt.traces:
            raise ValueError("No traces found in the filtered stream")

        trace_filt = stream_filt.traces[0].copy()
        trace_times_filt = trace_filt.times()
        trace_data_filt = trace_filt.data

        sta_lta_start = time.time()
        df = trace_filt.stats.sampling_rate
        sta_len, lta_len = 100, 1500
        cft = classic_sta_lta(trace_data_filt, int(sta_len * df), int(lta_len * df))
        metrics["sta_lta_time"] = time.time() - sta_lta_start

        analysis_start = time.time()
        local_maxima = find_local_maxima(cft)
        if len(local_maxima) == 0:
            raise ValueError("No local maxima found in the CFT")

        n_maxima = 10
        time_range = 200
        selected_maxima = select_top_maxima(trace_times_filt[local_maxima], cft[local_maxima], n_maxima, time_range)
        merged_rectangles, cft_max = get_data_regions(trace_times_filt, local_maxima[selected_maxima], cft)
        metrics["analysis_time"] = time.time() - analysis_start

        csv_start = time.time()
        df1 = pd.DataFrame({'CFT': cft, 'Time': trace_times_filt, 'Velocity': trace_data_filt})
        df2 = pd.DataFrame({'Start Time': [rect[0] for rect in merged_rectangles], 
                            'End time': [rect[1] for rect in merged_rectangles], 
                            'Percentages': [rect[2] for rect in merged_rectangles]})

        csv_buffer1 = io.StringIO()
        csv_buffer2 = io.StringIO()
        df1.to_csv(csv_buffer1, index=False)
        df2.to_csv(csv_buffer2, index=False)
        metrics["csv_time"] = time.time() - csv_start

        metrics["peak_memory"] = get_memory_usage()
        metrics["total_time"] = time.time() - metrics["start_time"]
        metrics["num_local_maxima"] = len(local_maxima)
        metrics["num_merged_regions"] = len(merged_rectangles)

        # Prepare the response JSON
        response_data = {
            "filename": file.filename,
            "cft_max": cft_max,
            "metrics": metrics,
            "data_csv": csv_buffer1.getvalue(),
            "indexes_csv": csv_buffer2.getvalue()
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        os.unlink(temp_file_path)

@app.get("/")
async def read_root():
    return {"message": "This API works"}