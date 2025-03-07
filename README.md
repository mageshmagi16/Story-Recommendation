# Pratilipi Recommendation System  

## README: Story Recommendation System using Neural Networks

**Overview:**
This project uses a neural network (PyTorch) to build a story recommendation system.
It predicts the most relevant stories for users based on their past reading behavior and category preferences.

### Installation & Setup
### 1. Clone the Repository
```
git clone https://github.com/Story-Recommendation/mageshmagi16.git
cd Story-Recommendation
```
### 2️. Create a Virtual Environment (Recommended)
```
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
Make sure you have Python 3.8+ installed. Then, install the required libraries:

**Data Handling**
```
import pandas as pd
import numpy as np
```

**Display and Visualization**
```
from IPython.display import display
import matplotlib.pyplot as plt
```

**Data Preprocessing and Normalization**
```
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```

**PyTorch for Deep Learning**
```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
```

**Model Evaluation Metrics**
```
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

### Running the Project

### 4. Load the Dataset

**Dataset Information:**

**1. `user_interaction.csv`**
This file contains user interactions with different pratilipis.

- **`user_id`** - Unique identifier for the user  
- **`pratilipi_id`** - Unique identifier for the story  
- **`read_percent`** - Percentage of the story read (0-100)  
- **`updated_at`** - Timestamp of the interaction 

**2. `metadata.csv`**
This file contains information about each pratilipi.

- **`author_id`** - Unique ID of the story's author  
- **`pratilipi_id`** - Unique identifier of the pratilipi  
- **`category_name`** - Category of the story (e.g., Romance, Social, Novels)  
- **`reading_time`** - Estimated reading time (in seconds). Assumes a reading speed of 200 words per minute  
- **`published_at`** - Timestamp when the story was published  

### 5. Data Preprocessing and Train the Model

Run the following script to load and clean the dataset, perform feature engineering, train the PyTorch-based neural network, generate top 5 recommended pratilipis per user.
```
Pratilipis_Recommendation_System.ipynb
```

### 6. Load Weights & Use the Trained Model
Another method is to directly load the weights. From this you can skip the training part and go for generating top 5 recommendations per user.
```
# Load model architecture and weights
model.load_state_dict(torch.load("full_model.pth"))
model.to(device)
```

## Project Files

| File | Description |
|------|------------|
| **`Pratilipis_Recommendation_System.ipynb`** | Data loading, preprocessing, model training, and saving weights. |
| **`full_model.pth`** | The saved entire model (architecture + weights). |
| **`Pratilipis_Recommendation_System.pdf`** | Documentation of the Pratilipis Recommendation System. |
| **`full_model.png`** | Entire model - Image. |










