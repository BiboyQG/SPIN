from pydantic import BaseModel, Field
from typing import List, Union

class GPUEngineSpecs(BaseModel):
    cuda_cores: int = Field(..., alias="NVIDIACUDACores")
    shader_cores: str = Field(..., alias="ShaderCores")
    ray_tracing_cores: str = Field(..., alias="RayTracingCores")
    tensor_cores: str = Field(..., alias="TensorCores")
    boost_clock: float = Field(..., alias="BoostClock")
    base_clock: float = Field(..., alias="BaseClock")

class MemorySpecs(BaseModel):
    standard_memory_config: str = Field(..., alias="StandardMemoryConfig")
    memory_interface_width: int = Field(..., alias="MemoryInterfaceWidth")

class TechnologySupport(BaseModel):
    nvidia_architecture: str = Field(..., alias="NVIDIAArchitecture")
    ray_tracing: bool = Field(..., alias="RayTracing")
    nvidia_dlss: List[str] = Field(..., alias="NVIDIADLSS")
    nvidia_reflex: bool = Field(..., alias="NVIDIAReflex")
    nvidia_broadcast: bool = Field(..., alias="NVIDIABroadcast")
    pci_express_gen_4: bool = Field(..., alias="PCIExpressGen4")
    resizable_bar: bool = Field(..., alias="ResizableBAR")
    nvidia_geforce_experience: bool = Field(..., alias="NVIDIAGeForceExperience")
    nvidia_ansel: bool = Field(..., alias="NVIDIAAnsel")
    nvidia_freestyle: bool = Field(..., alias="NVIDIAFreeStyle")
    nvidia_shadowplay: bool = Field(..., alias="NVIDIAShadowPlay")
    nvidia_highlights: bool = Field(..., alias="NVIDIAHighlights")
    nvidia_g_sync: bool = Field(..., alias="NVIDIAGSYNC")
    game_ready_drivers: bool = Field(..., alias="GameReadyDrivers")
    nvidia_studio_drivers: bool = Field(..., alias="NVIDIAStudioDrivers")
    nvidia_omniverse: bool = Field(..., alias="NVIDIAOmniverse")
    microsoft_directx_12_ultimate: bool = Field(..., alias="MicrosoftDirectX12Ultimate")
    nvidia_gpu_boost: bool = Field(..., alias="NVIDIAGPUBOOST")
    nvidia_nvlink: bool = Field(..., alias="NVIDIANVLink")
    vulkan_rt_api: bool = Field(..., alias="VulkanRTAPI")
    nvidia_encoder: str = Field(..., alias="NVIDIAEncoder")
    nvidia_decoder: str = Field(..., alias="NVIDIADecoder")
    av1_encode: bool = Field(..., alias="AV1Encode")
    av1_decode: bool = Field(..., alias="AV1Decode")
    cuda_capability: float = Field(..., alias="CUDACapability")
    vr_ready: bool = Field(..., alias="VRReady")

class DisplaySupport(BaseModel):
    maximum_resolution: str = Field(..., alias="MaximumResolution")
    standard_display_connectors: str = Field(..., alias="StandardDisplayConnectors")
    multi_monitor: str = Field(..., alias="MultiMonitor")
    hdcp: str = Field(..., alias="HDCP")

class CardDimensions(BaseModel):
    length: int = Field(..., alias="Length")
    width: int = Field(..., alias="Width")
    slots: str = Field(..., alias="Slots")

class ThermalAndPowerSpecs(BaseModel):
    maximum_gpu_temperature: int = Field(..., alias="MaximumGPUTemperature")
    idle_power: int = Field(..., alias="IdlePower")
    video_playback_power: int = Field(..., alias="VideoPlaybackPower")
    average_gaming_power: int = Field(..., alias="AverageGamingPower")
    total_graphics_power: int = Field(..., alias="TotalGraphicsPower")
    required_system_power: int = Field(..., alias="RequiredSystemPower")
    supplementary_power_connectors: str = Field(..., alias="SupplementaryPowerConnectors")

class GraphicsCard(BaseModel):
    name: str
    description: str
    gpu_engine_specs: GPUEngineSpecs = Field(..., alias="GPUEngineSpecs")
    memory_specs: MemorySpecs = Field(..., alias="MemorySpecs")
    technology_support: TechnologySupport = Field(..., alias="TechnologySupport")
    display_support: DisplaySupport = Field(..., alias="DisplaySupport")
    card_dimensions: CardDimensions = Field(..., alias="CardDimensions")
    thermal_and_power_specs: ThermalAndPowerSpecs = Field(..., alias="ThermalAndPowerSpecs")

    class Config:
        allow_population_by_field_name = True
        alias_generator = lambda field_name: "".join(
            word.capitalize() if i > 0 else word
            for i, word in enumerate(field_name.split("_"))
        )