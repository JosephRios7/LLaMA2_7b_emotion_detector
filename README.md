# LLaMA 2 7B — Emotion Detector (Spanish)

<p align="center">
  <img src="https://img.shields.io/badge/Precisión-89.04%25-brightgreen"/>
  <img src="https://img.shields.io/badge/F1--Score_Macro-89.02%25-brightgreen"/>
  <img src="https://img.shields.io/badge/Idioma-Español-blue"/>
  <img src="https://img.shields.io/badge/Modelo_Base-LLaMA_2_7B-orange"/>
  <img src="https://img.shields.io/badge/Técnica-LoRA_+_4bit-purple"/>
  <img src="https://img.shields.io/badge/Hardware-Kaggle_P100-lightgrey"/>
</p>

Repositorio del Trabajo de Integración Curricular:
**"Ajuste fino del modelo LLaMA 2 para la detección de emociones 
en reflexiones personales en español"**

Universidad Nacional de Loja — Carrera de Computación — 2025

---

## Descripción

Este proyecto aplica ajuste fino (*fine-tuning*) al modelo LLaMA 2 7B 
para clasificar emociones en textos en español. El modelo detecta seis 
emociones: **ira, disgusto, tristeza, alegría, miedo y neutral**.

La motivación principal es reducir la brecha tecnológica entre el español 
y el inglés en el campo del análisis afectivo automatizado, con aplicaciones 
en salud mental, educación y psicología clínica.

---

## Resultados

| Métrica | Valor |
|---|---|
| Precisión global | **89,04%** |
| F1-score macro | **89,02%** |
| Conjunto de prueba | 3.366 muestras balanceadas |
| Tiempo de inferencia (T4) | ~1,25 seg/muestra |

### Por emoción

| Emoción | Precisión | Recall | F1-Score |
|---|---|---|---|
| Disgusto | 0,9514 | 0,9412 | **0,9462** |
| Miedo | 0,9475 | 0,9323 | **0,9398** |
| Neutral | 0,8318 | 0,9430 | 0,8839 |
| Alegría | 0,8891 | 0,8574 | 0,8730 |
| Tristeza | 0,8623 | 0,8485 | 0,8553 |
| Ira | 0,8679 | 0,8200 | 0,8433 |

---

## Estructura del repositorio
```
LLaMA2_7b_emotion_detector/
│
├── notebooks/
│   ├── 01_dataset_cleaning.ipynb        # Limpieza y preprocesamiento
│   ├── 02_dataset_balancing.ipynb       # Balanceo del dataset
│   ├── 03_finetuning_v9.ipynb           # Entrenamiento del modelo V9
│   ├── 04_postprocessing.ipynb          # Fusión de pesos y publicación
│   └── 05_evaluation.ipynb             # Evaluación y métricas finales
│
├── data/
│   └── README.md                        # Descripción del dataset
│
└── README.md
```

> Los notebooks de entrenamiento están diseñados para ejecutarse en 
> **Kaggle** con GPU P100. Los notebooks de evaluación funcionan en 
> **Google Colab** con GPU T4.

---

## Dataset

Dataset unificado de **33.672 textos balanceados** construido a partir 
de cuatro corpus de referencia:

| Corpus | Muestras | Idioma original |
|---|---|---|
| Spanish MEACorpus 2023 | 5.129 | Español |
| EmoEvalEs 2021 | 8.409 | Español |
| CARER Emotion Dataset | 20.000 | Inglés → traducido |
| HRECPW Dataset | 125.217 | Inglés → traducido |

Disponible en Hugging Face:
👉 [Joseph7D/emotion-dataset-v2](https://huggingface.co/datasets/Joseph7D/emotion-dataset-v2)

**Validación externa:** psicólogo clínico certificó una validez del 
**98,67%** sobre una muestra de 300 textos (50 por emoción).

---

## Modelo

Disponible en Hugging Face:
👉 [Joseph7D/llama-2-7b-emotion-detector-v9](https://huggingface.co/Joseph7D/llama-2-7b-emotion-detector-v9)

### Uso rápido
```python
# Celda 1 — Instalar dependencias
!pip install accelerate peft bitsandbytes transformers gradio

# Celda 2 — Imports
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Celda 3 — Cargar modelo
model_id = "Joseph7D/llama-2-7b-emotion-detector-v9"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Celda 4 — Función de clasificación
def detectar_emocion(texto):
    model.eval()
    prompt = (
        f"Clasifica el siguiente texto en una de estas emociones: "
        f"ira, disgusto, tristeza, alegría, miedo o neutral. "
        f"Responde únicamente con la emoción correspondiente.\n\n"
        f"Texto: \"{texto}\""
    )
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        return_attention_mask=True
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=20,
            do_sample=False
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("[/INST]")[-1].strip()

# Ejemplos
print(detectar_emocion("Me siento aterrado cada vez que tengo que hablar frente a mi jefe"))
# Output: miedo

print(detectar_emocion("Estoy tan feliz de haber terminado este proyecto"))
# Output: alegría

print(detectar_emocion("Me enfurece la impuntualidad y la falta de compromiso"))
# Output: ira
```

---

## Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Modelo base | clibrain/Llama-2-7b-ft-instruct-es |
| Técnica | LoRA (Low-Rank Adaptation) |
| Rank (r) | 8 |
| Alpha | 32 |
| Dropout | 0,10 |
| Cuantización entrenamiento | 4-bit NF4 |
| Cuantización inferencia | 8-bit |
| Epochs | 2 |
| Learning rate | 2e-5 |
| Batch efectivo | 8 (batch=1 × grad_accum=8) |
| Optimizer | paged_adamw_32bit |
| Weight decay | 0,01 |
| Warmup ratio | 10% |
| Scheduler | Linear |
| Max grad norm | 0,5 |
| Hardware | GPU NVIDIA Tesla P100 16GB (Kaggle) |
| Tiempo de entrenamiento | 9h 39min |

---

## Metodología

El proyecto siguió la metodología **CRISP-ML(Q)** adaptada a las 
particularidades del proyecto, en cinco fases:

1. **Comprensión del problema y los datos** — análisis de corpus y definición de emociones objetivo
2. **Preparación de datos** — limpieza, traducción, normalización y balanceo
3. **Ajuste fino del modelo** — aplicación de LoRA e iteración de hiperparámetros (V4→V9)
4. **Post-procesamiento** — fusión de pesos, cuantización y publicación
5. **Evaluación** — métricas de clasificación sobre conjunto de prueba independiente

---

## Comparación con el estado del arte

| Trabajo | Modelo | Resultado |
|---|---|---|
| EmoLLMs (TR01) | EmoLLaMA-7B | 54,50% |
| MEACorpus (TR03) | MarIA (BERT-es) | 69,39% |
| GPT-4 zero-shot (TR06) | GPT-4 | 82,20% |
| EmoBERTTiny (TR04) | BERT compacto | 85,46% |
| **Este trabajo** | **LLaMA 2 7B + LoRA** | **89,04%** |

---

## Limitaciones

- Predominio del **español peninsular** en los corpus de entrenamiento.
- Dificultad con **ironía y sarcasmo**.
- Diseñado para textos entre 3 y 200 palabras.
- Evaluado solo en las seis emociones definidas.

---

## Cita
**BibTeX:**
```bibtex
@thesis{rios2025llama2emotions,
  title     = {Ajuste fino del modelo LLaMA 2 para la detección de 
               emociones en reflexiones personales en español},
  author    = {Ríos Salas, Joseph Daniel},
  year      = {2025},
  school    = {Universidad Nacional de Loja},
  type      = {Trabajo de Integración Curricular},
  url       = {https://dspace.unl.edu.ec/items/396f65b5-5e5e-4d68-bc52-9999fc790c90}
}
```

**APA:**
Ríos Salas, J. D. (2025). *Ajuste fino del modelo LLaMA 2 para la 
detección de emociones en reflexiones personales en español* 
[Trabajo de Integración Curricular, Universidad Nacional de Loja]. 
Repositorio Institucional UNL. 
https://dspace.unl.edu.ec/items/396f65b5-5e5e-4d68-bc52-9999fc790c90

---

## Autor

**Joseph Daniel Ríos Salas**  
Universidad Nacional de Loja — Carrera de Computación  
Director: Ing. Oscar Miguel Cumbicus Pineda, Mg.Sc.  
📧 joseph.rios@unl.edu.ec  
🤗 [Hugging Face](https://huggingface.co/Joseph7D)  
🐙 [GitHub](https://github.com/JosephRios7)
