# Getting Started - Visual Guide
## Follow This Path to Success! 🎯

```
START HERE
    ↓
┌──────────────────────────────────────┐
│  Step 1: Install Dependencies        │
│  Command: pip install -r             │
│           requirements.txt           │
│  Time: 3-5 minutes                   │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  Step 2: Setup MLflow                │
│  Command: python scripts/            │
│           setup_mlflow.py            │
│  Time: 1 minute                      │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  Step 3: Create Training Data        │
│  Command: python setup_data.py       │
│  Time: 1-2 minutes                   │
└──────────────────────────────────────┘
    ↓
    ├─────────────────────────────────────────────┐
    │                                             │
    ↓                                             ↓
┌──────────────────────────┐        ┌──────────────────────────┐
│  Option A: NER Model     │        │  Option B: Classifier    │
│  python scripts/         │        │  python scripts/         │
│    train_ner.py          │        │    train_classifier.py   │
│  Time: 5-10 min          │        │  Time: 5-10 min          │
└──────────────────────────┘        └──────────────────────────┘
    │                                             │
    └─────────────────┬───────────────────────────┘
                      ↓
┌──────────────────────────────────────┐
│  Step 5: View Results                │
│  Open: http://localhost:5000         │
│  (MLflow UI)                         │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  SUCCESS! 🎉                         │
│  You've trained your first model!    │
└──────────────────────────────────────┘
```

---

## 📋 Command Cheat Sheet

### The Essential 3 Commands

```bash
# 1. Install everything
pip install -r requirements.txt

# 2. Create training data
python setup_data.py

# 3. Train your model
python scripts/train_ner.py --data-path data/processed --use-mlflow
```

That's it! You're done! 🎉

---

## 🎓 What Each Step Does

### Step 1: Install Dependencies
**What it does:** Downloads and installs all required Python packages
**Why it's needed:** You need these libraries to run the code
**How long:** 3-5 minutes
**Sign of success:** No error messages, ends with "Successfully installed..."

### Step 2: Setup MLflow
**What it does:** Creates folders and sets up experiment tracking
**Why it's needed:** To save and visualize your training results
**How long:** 1 minute
**Sign of success:** You see "✓ MLflow setup completed successfully!"

### Step 3: Create Training Data
**What it does:** Generates 100 fake medical text examples
**Why it's needed:** You need data to train the model
**How long:** 1-2 minutes
**Sign of success:** You see "✅ Data Setup Complete!"

### Step 4: Train Model
**What it does:** Teaches the AI to recognize medical entities
**Why it's needed:** This is the main goal - creating a trained model!
**How long:** 5-10 minutes
**Sign of success:** You see "Training completed!" and metrics are printed

### Step 5: View Results
**What it does:** Shows graphs and metrics of your trained model
**Why it's needed:** To see how well your model performed
**How long:** 30 seconds
**Sign of success:** You see the MLflow web interface with your experiment

---

## 🎯 Decision Tree: Which Model Should I Train?

```
Start: What do you want to do?
    │
    ├─ Extract medical terms from text (diseases, drugs, etc.)
    │  → Train NER Model (Step 4A)
    │  → Use: python scripts/train_ner.py
    │
    ├─ Classify medical documents into categories
    │  → Train Classifier (Step 4B)
    │  → Use: python scripts/train_classifier.py
    │
    └─ Try both
       → Train NER first, then Classifier
       → Takes about 10-20 minutes total
```

---

## 🆘 Troubleshooting Quick Guide

### Problem: "pip: command not found"
```bash
# Solution: Use python -m pip instead
python -m pip install -r requirements.txt
```

### Problem: "ImportError: No module named..."
```bash
# Solution: Install the specific package
pip install [package-name]
```

### Problem: Training is too slow
```bash
# Solution: Use smaller settings
python scripts/train_ner.py --batch-size 4 --epochs 2
```

### Problem: Out of memory
```bash
# Solution: Reduce batch size
python scripts/train_ner.py --batch-size 4
```

### Problem: "data/processed not found"
```bash
# Solution: Run the data setup first
python setup_data.py
```

---

## 📊 What Your Training Output Means

When you train a model, you'll see something like this:

```
Epoch 1/3: 100%|████████████| Loss: 0.543
Validation F1: 0.756
✓ New best model saved!
```

**What this means:**
- **Epoch 1/3**: First pass through the data (out of 3 total)
- **Loss: 0.543**: How wrong the model is (lower = better)
- **Validation F1: 0.756**: Accuracy score (higher = better, max = 1.0)
- **New best model saved**: This is the best result so far, saving it!

**Good scores:**
- F1 > 0.7 → Pretty good! ✅
- F1 > 0.8 → Great! 🎉
- F1 > 0.9 → Excellent! 🌟

---

## 🎯 Your Goal

```
┌─────────────────────────────────────────────┐
│  After following this guide, you will have: │
├─────────────────────────────────────────────┤
│  ✅ Working Python environment              │
│  ✅ All dependencies installed              │
│  ✅ Training data ready                     │
│  ✅ At least one trained model              │
│  ✅ Ability to view results in MLflow       │
│  ✅ Understanding of the workflow           │
└─────────────────────────────────────────────┘
```

---

## 📚 Next Steps After Basic Training

Once you've successfully trained your first model:

1. **Experiment with Settings**
   - Try different learning rates: `--lr 1e-5` or `--lr 3e-5`
   - Change batch size: `--batch-size 16`
   - More epochs: `--epochs 10`

2. **Train the Other Model**
   - If you trained NER, try the classifier
   - If you trained classifier, try NER

3. **Use Real Data**
   - See COMPLETE_SETUP_GUIDE.md, Section "Option B"
   - Download real medical datasets
   - Format them correctly

4. **Deploy Your Model**
   - Register in MLflow: `python scripts/model_registry.py register`
   - Start the API: `python -m uvicorn src.api.main:app`
   - Make predictions via HTTP requests

5. **Monitor Performance**
   - Track metrics: `python scripts/monitor_models.py`
   - Generate reports
   - Detect data drift

---

## 🎓 Learning Resources

- **[COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)**: Full detailed guide
- **[QUICKSTART.md](QUICKSTART.md)**: Quick reference
- **[docs/mlops.md](docs/mlops.md)**: Advanced MLOps workflows
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**: Technical documentation

---

## ✅ Pre-Flight Checklist

Before you start, verify:
- [ ] Python 3.8+ is installed (`python --version`)
- [ ] You're in the project directory (`cd medical-ai-assistant`)
- [ ] You have internet connection (to download models)
- [ ] You have 4GB+ free disk space

---

## 🎉 Success Indicators

You'll know you're successful when:
1. ✅ `pip install` completes without errors
2. ✅ `setup_data.py` prints "Data Setup Complete!"
3. ✅ Training shows decreasing loss values
4. ✅ You see F1 scores > 0.7
5. ✅ MLflow UI opens and shows your experiment
6. ✅ Model files are saved in `models/` folder

---

## 💡 Pro Tips

1. **Start with defaults**: Don't change settings until you've run it once successfully
2. **Be patient**: First training takes longer (downloads models)
3. **Check MLflow UI**: It's the best way to see what's happening
4. **Save your commands**: Keep a text file of commands that worked
5. **Read error messages**: They usually tell you exactly what's wrong

---

**Ready to start? Go to → [COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)**

**Have 5 minutes? Try → [QUICKSTART.md](QUICKSTART.md)**

**Need help? Check troubleshooting in [COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)**
