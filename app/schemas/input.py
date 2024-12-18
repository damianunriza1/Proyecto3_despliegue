# model/processing/input.py
from typing import Optional
from pydantic import BaseModel

class Input(BaseModel):
    FID_puntos_comerciales: Optional[int]
    municipio: Optional[str]
    departamen: Optional[str]
    cod_depart: Optional[int]
    cod_dane_mpio: Optional[int]
    categoria_mtv11: Optional[str]
    cod_precios: Optional[str]
    Vigencia: Optional[int]
    FID_ECOSISTEMAS_2017: Optional[int]
    OBJECTID_1: Optional[int]
    TIPO_ECOSI: Optional[str]
    GRADO_TRAN: Optional[str]
    GRAN_BIOMA: Optional[str]
    BIOMA_PREL: Optional[str]
    BIOMA_IAvH: Optional[str]
    ECOS_SINTE: Optional[str]
    ECOS_GENER: Optional[str]
    UNIDAD_SIN: Optional[str]
    AMBIENTE_A: Optional[str]
    SUBSISTEMA: Optional[str]
    ZONA_HIDRO: Optional[str]
    ORIGEN: Optional[str]
    TIPO_AGUA: Optional[str]
    CLIMA: Optional[str]
    PAISAJE: Optional[str]
    RELIEVE: Optional[str]
    SUELOS: Optional[str]
    AMB_EDAFOG: Optional[str]
    DESC_AMB_E: Optional[str]
    COBERTURA: Optional[str]
    UNI_BIOTIC: Optional[str]
    ANFIBIOS: Optional[str]
    AVES: Optional[str]
    MAGNOLIOPS: Optional[str]
    MAMIFEROS: Optional[str]
    REPTILES: Optional[str]
    No_Anfibio: Optional[int]
    No_Aves: Optional[int]
    No_Magnoli: Optional[int]
    No_Mamifer: Optional[int]
    No_Reptile: Optional[int]
    SHAPE_Leng: Optional[float]
    AH: Optional[int]
    NOMAH: Optional[str]
    ZH: Optional[int]
    NOMZH: Optional[str]
    SZH: Optional[int]
    NOMSZH: Optional[str]
    IPHE_VALOR: Optional[float]
    IPHE_CATEGORIA: Optional[str]
    IEUA_VALOR: Optional[float]
    IUEA_CATEGORIA: Optional[str]
    HHV2020: Optional[float]
    HHA_2020: Optional[float]
    IARC_VALOR: Optional[float]
    IARC_CATEGORIA: Optional[str]
    IUA_MEDIO_VALOR: Optional[float]
    IUA_MEDIO_CAREGORIA: Optional[str]
    IUA_SECO_VALOR: Optional[float]
    IUA_sECO_CATEGORIA: Optional[str]
    IACAL_M: Optional[float]
    CATIACAL_M: Optional[str]
    IACAL_S: Optional[float]
    CIACAL_S: Optional[str]
    irh: Optional[float]
    cat: Optional[str]
    IVH_MEDIO_VALOR: Optional[float]
    IVH_MEDIO_CATEGORIA: Optional[str]
    IVH_SECO_VALOR: Optional[float]
    IVH_SECO_CATEGORIA: Optional[str]
    FID_Correlacion_5048_md: Optional[int]
    UCSuelo: Optional[str]
    S_UC: Optional[str]
    S_UC_FASE: Optional[str]
    OTRAS_FA_1: Optional[str]
    PENDIENT_1: Optional[str]
    EROSION_1: Optional[str]
    S_CLIMA: Optional[str]
    CLIMA_1: Optional[str]
    S_PAISAJE: Optional[str]
    PAISAJE_1: Optional[str]
    TIPO_RELIE: Optional[str]
    MATERIAL_P: Optional[str]
    SUBGRUPO: Optional[str]
    TEXTURA: Optional[str]
    CLAS_UPRA: Optional[str]
    area_Ha_12: Optional[float]
    FID_1: Optional[int]
    PORCENTAJE: Optional[float]
    HUMEDAD: Optional[str]
    leyenda: Optional[str]
    AF: Optional[str]
    PERFILES: Optional[str]
    SATURACION: Optional[str]
    DRENAJE: Optional[str]
    SALINIDAD_1: Optional[str]
    CALCAREO: Optional[str]
    ORIG_FID_1: Optional[int]
    SODICIDAD: Optional[str]
    ALUMINIO: Optional[str]
    ACIDEZ: Optional[str]
    Significad: Optional[str]
    FID_Cobertura_tierra_2018: Optional[int]
    FA: Optional[str]
    codigo: Optional[str]
    FERTILIDAD: Optional[str]
    COBERTURAS_CON_USO_AGRO: Optional[str]
    AREA_HA_1: Optional[float]
    layer: Optional[str]
    PROFUNDIDA: Optional[float]