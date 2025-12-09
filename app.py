from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from backend import PCBDefectDetector
import json

app = FastAPI(title="PCB Defect Detection API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = PCBDefectDetector()

@app.get("/")
async def root():
    return {"message": "PCB Defect Detection API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/api/detect")
async def detect_defects(
    template: UploadFile = File(...),
    test: UploadFile = File(...)
):
    """Process PCB images and detect defects"""
    try:
        # Validate file types
        if not template.content_type.startswith("image/"):
            raise HTTPException(400, "Template must be an image")
        if not test.content_type.startswith("image/"):
            raise HTTPException(400, "Test image must be an image")
        
        # Read uploaded files
        template_data = await template.read()
        test_data = await test.read()
        
        # Process images
        result = detector.process_request(template_data, test_data)
        
        if "error" in result:
            raise HTTPException(500, result["error"])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.post("/api/batch")
async def batch_process(
    template: UploadFile = File(...),
    tests: list[UploadFile] = File(...)
):
    """Process multiple test images with same template"""
    results = []
    
    template_data = await template.read()
    
    for test_file in tests:
        test_data = await test_file.read()
        result = detector.process_request(template_data, test_data)
        results.append(result)
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
