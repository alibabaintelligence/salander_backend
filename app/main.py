from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
   
app = FastAPI()
   
# Add CORS middleware
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],  # Allows all origins for prototype
   allow_credentials=True,
   allow_methods=["*"],  # Allows all methods
   allow_headers=["*"],  # Allows all headers
)
   
@app.get("/")
async def read_root():
   return {"message": "This works"}
   
@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):
    contents = await file.read()
    # Process the file contents here
    # For this example, we'll just return the file size
    return {"filename": file.filename, "size": len(contents)}
