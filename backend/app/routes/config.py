from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services import state

router = APIRouter()

VALID_MODES = {"instruct", "ragnarok_tuned", "local_ragnarok"}


class ConfigUpdate(BaseModel):
    model_mode: str


@router.get("/config")
async def get_config():
    """Return the current model mode."""
    return {"model_mode": state.model_mode}


@router.post("/config")
async def set_config(update: ConfigUpdate):
    """Switch between Instruct model and Ragnarok fine-tuned model."""
    if update.model_mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Choose from: {VALID_MODES}"
        )
    state.model_mode = update.model_mode
    return {"model_mode": state.model_mode}
