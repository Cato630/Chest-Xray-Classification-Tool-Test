# Chest X-Ray Classification Assistant

## Overview

The goal of this project was to build a simple web application that can classify a chest X-ray image as either **Normal** or **Abnormal** using a Vision Transformer (ViT). This project was completed as part of my Master's program in Artificial Intelligence in Medicine.

I wanted to build something that was a little more than just training a model in a notebook. Instead, I wanted to understand what the complete workflow looked like, starting with preparing a dataset, training a model, testing it, and then building a simple application that someone could actually interact with.

This is still a learning project, and I'm sure there are plenty of things that could be improved, but I learned a lot about how machine learning projects are structured from beginning to end.

---

## Features

- Upload a chest X-ray image
- Classify the image as **Normal** or **Abnormal**
- Display prediction confidence
- Download a prediction report
- Built using Gradio for a simple web interface
- Uses a Vision Transformer (ViT) model fine-tuned on the NIH Chest X-ray dataset

---

## Dataset

This project uses the **NIH ChestX-ray14** dataset.

The dataset itself is **not included** in this repository because of its size. The images and metadata can be downloaded directly from the NIH website.

Dataset:
https://nihcc.app.box.com/v/ChestXray-NIHCC

---

## Project Structure

```text
clinical-radiology-classifier/

│
├── app.py
├── inference.py
├── prepare_dataset.py
├── train_model.py
├── test_inference.py
├── requirements.txt
├── README.md
```

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Gradio
- Pandas
- NumPy
- Pillow
- Scikit-learn

---

## Running the Project

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

The application will start locally and provide a URL that can be opened in your browser.

---

## What I Learned

I ended up learning a lot more about myself and ML during this project than I expected.

When I first started, I honestly had no idea how machine learning projects were organized outside of a Jupyter Notebook. Building this forced me to separate different parts of the project into individual files, learn how to save and reload trained models, create reusable functions, and build a simple user interface.

I also learned a lot about debugging. Most of the time wasn't spent writing code—it was figuring out why something didn't work and then learning enough to fix it.

Even though this is still a relatively simple application, I have a much better understanding of what an end-to-end machine learning project looks like now.

---

## Future Improvements

Some ideas I'd like to continue working on after this class include:

- Support for additional chest diseases instead of only Normal/Abnormal
- Better visualization of model predictions
- Improved user interface
- Deploying the application publicly using Hugging Face Spaces
- Experimenting with larger datasets and different transformer models

---

## Disclaimer

This application was created for educational purposes as part of a graduate course.

It is **not** intended to provide medical advice, diagnosis, or treatment decisions.
