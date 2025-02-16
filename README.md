# 3D_datagen

A framework for generating and processing synthetic 3D data using **Infinigen, Zeroverse, and Shapegen**.

---

## 🚀 Clone this Repository
```bash
git clone https://github.com/VaclavKnapp/3D_datagen.git
```

---

## 🐍 create conda environments
```bash
conda env create -f 3D_datagen/infinigen/env.yml
conda env create -f 3D_datagen/zeroverse/env.yml
conda env create -f 3D_datagen/shapegen/env.yml
```
---

## Infinigen

### Activate env
```bash
conda activate infinigen
```

### Clone the Infinigen Repository
```bash
git clone https://github.com/princeton-vl/infinigen.git
```

### Move Required Files
```bash
mv 3D_datagen/infinigen/gen_assests_blend.py infinigen/infinigen_examples
mv 3D_datagen/infinigen/oddity.py infinigen/.
mv 3D_datagen/infinigen/render.sh infinigen/.
```

### 📸 Generating Images

#### 1️⃣ Generate .blend Files
```bash
cd infinigen
python infinigen_examples/gen_assests_blend.py -f 3D_datagen/infinigen/factories.txt -o 3D_datagen/infinigen_blend -n 10 --texture_folder 3D_datagen/textures
```

#### 2️⃣ Render Images
```bash
bash infinigen/render.sh 3D_datagen/infinigen_blend 3D_datagen/infinigen_images 3D_datagen/backgrounds
```

### 🔍 Running Oddity Detection
1. Modify **line 203** in `infinigen/oddity.py` to include the generated images path (e.g., `'3D_datagen/infinigen_images'`).
2. Run the script:
```bash
python infinigen/oddity.py
```

---

## Zeroverse (My Implementation)

### Activate env
```bash
conda activate zeroverse
```

### Configure Output Folder
Edit `generate_sets.sh` to specify your desired output directory on line `3`.

### 📸 Generating and Rendering Images
#### 1️⃣ Generate .blend Files
```bash
bash 3D_datagen/zeroverse/generate_sets.sh
```

#### 2️⃣ Render Images
```bash
bash 3D_datagen/zeroverse/render.sh YOUR_BLEND_FOLDER 3D_datagen/zeroverse_images 3D_datagen/backgrounds
```

### 🔍 Running Oddity Detection
1. Modify **line 258** in `zeroverse/oddity.py` to include the absolute path of the generated images (e.g., `'3D_datagen/zeroverse_images'`).
2. Run the script:
```bash
python infinigen/oddity.py
```

---

## Shapegen

### Activate env
```bash
conda activate shapegen
```

### 📸 Generating and Rendering Images

#### 1️⃣ Generate .blend Files
This will generate 10 blocky and 10 smooth objects per `n_extrusions`.
```bash
python 3D_datagen/shapegen/generate_objects.py -- -o 3D_datagen/shapegen/blend_files -n_extrusions 3
python 3D_datagen/shapegen/generate_objects.py -- -o 3D_datagen/shapegen/blend_files -n_extrusions 4
python 3D_datagen/shapegen/generate_objects.py -- -o 3D_datagen/shapegen/blend_files -n_extrusions 5
python 3D_datagen/shapegen/generate_objects.py -- -o 3D_datagen/shapegen/blend_files -n_extrusions 6
python 3D_datagen/shapegen/generate_objects.py -- -o 3D_datagen/shapegen/blend_files -n_extrusions 7
python 3D_datagen/shapegen/generate_objects.py -- -o 3D_datagen/shapegen/blend_files -n_extrusions 8
```

#### 2️⃣ Render Images
```bash
bash 3D_datagen/shapegen/render.sh 3D_datagen/shapegen/blend_files 3D_datagen/shapegen/images 3D_datagen/backgrounds
```

#### 3️⃣ Measure Oddity Accuracy
1. Modify **line 171** in `shapegen/oddity.py` to include the absolute path of the generated images (e.g., `3D_datagen/shapegen/images`).
2. Run the script:
```bash
python 3D_datagen/shapegen/oddity.py
```



