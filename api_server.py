from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from src.agent.workflow import app as agent_app

api = FastAPI(title="Automotive Diagnostic Agent API", version="3.0")

class VehicleContext(BaseModel):
    brand: str = "Toyota"
    model: str = "Camry"
    year: str = "2020"
    engine: str = "2.5L"

class DiagnoseRequest(BaseModel):
    question: str
    vehicle_info: VehicleContext = VehicleContext()

@api.post("/api/v1/diagnose")
def diagnose_vehicle(req: DiagnoseRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="提问内容不能为空")

    inputs = {
        "messages": [HumanMessage(content=req.question)],
        "vehicle_info": req.vehicle_info.model_dump()
    }

    result = agent_app.invoke(inputs)
    final_output = result["messages"][-1].content

    return {
        "question": req.question,
        "vehicle_info": req.vehicle_info,
        "report": final_output
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:api", host="0.0.0.0", port=8000, reload=True)