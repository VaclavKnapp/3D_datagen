import os
import random
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns  

# pip install torch torchvision numpy matplotlib scikit-learn seaborn tqdm

def load_dino_v2_model():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval()
    return model

def extract_feature(model, image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        feature = model(input_tensor)
    return feature.cpu().squeeze().numpy()

transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def compute_similarity(features):
    return cosine_similarity(features)

def oddity_detection(object_A, object_B, model, transform, device, base_path, trial_num=None, trial=None):
    images_A = random.sample(object_A, 2)
    image_B = random.choice(object_B)
    image_paths = images_A + [image_B]

    random.shuffle(image_paths)
    b_index = image_paths.index(image_B)

    features = [extract_feature(model, img_path, transform, device) for img_path in image_paths]

    similarity_matrix = compute_similarity(features)

    off_diag_corr_A = (similarity_matrix[0, 1] + similarity_matrix[0, 2]) / 2
    off_diag_corr_A_prime = (similarity_matrix[1, 0] + similarity_matrix[1, 2]) / 2
    off_diag_corr_B = (similarity_matrix[2, 0] + similarity_matrix[2, 1]) / 2

    correlations = [off_diag_corr_A, off_diag_corr_A_prime, off_diag_corr_B]
    b_index_guess = np.argmin(correlations)

    is_correct = b_index_guess == b_index


    if trial_num is not None and trial_num % 5 == 0:

        plot_dir = os.path.join(base_path, 'plots', f'{trial}')
        os.makedirs(plot_dir, exist_ok=True)
        labels = ['Image 1', 'Image 2', 'Image 3']

        plt.figure(figsize=(6, 5))
        sns.heatmap(similarity_matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap='viridis')
        plt.title(f'Similarity Matrix - Trial {trial_num}')
        plt.tight_layout()
        heatmap_filename = os.path.join(plot_dir, f'similarity_heatmap_trial_{trial_num}.png')
        plt.savefig(heatmap_filename, dpi=300)
        plt.close()

        images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
        images = [img.resize((256, 256)) for img in images]

        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        combined_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        x_positions = []
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_positions.append(x_offset)
            x_offset += img.width

        draw = ImageDraw.Draw(combined_image)
        border_color_correct = (0, 255, 0)
        border_color_incorrect = (255, 0, 0)
        border_width = 5

        y0 = 0
        y1 = images[0].height


        x0 = x_positions[b_index]
        x1 = x_positions[b_index] + images[b_index].width
        draw.rectangle([x0, y0, x1, y1], outline=border_color_correct, width=border_width)


        if b_index_guess != b_index:
            x0 = x_positions[b_index_guess]
            x1 = x_positions[b_index_guess] + images[b_index_guess].width
            draw.rectangle([x0, y0, x1, y1], outline=border_color_incorrect, width=border_width)


        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()


        text_correct = 'Correct Odd Image'
        text_width, text_height = draw.textsize(text_correct, font=font)
        x_text = x_positions[b_index] + images[b_index].width / 2 - text_width / 2
        y_text = y0 - text_height - 5
        if y_text < 0:
            y_text = y1 + 5
        draw.text((x_text, y_text), text_correct, fill=(0, 255, 0), font=font)


        if b_index_guess != b_index:
            text_guess = 'Model Guess'
            fill_color = (255, 0, 0)
            x_text = x_positions[b_index_guess] + images[b_index_guess].width / 2 - text_width / 2
            y_text = y0 - text_height - 5
            if y_text < 0:
                y_text = y1 + 5
            draw.text((x_text, y_text), text_guess, fill=fill_color, font=font)
        else:
            text_guess = 'Model Correct'
            fill_color = (0, 255, 0)
            x_text = x_positions[b_index_guess] + images[b_index_guess].width / 2 - text_width / 2
            y_text = y0 - text_height - 5
            if y_text < 0:
                y_text = y1 + 5
            draw.text((x_text, y_text), text_guess, fill=fill_color, font=font)


        triplet_filename = os.path.join(plot_dir, f'triplet_trial_{trial_num}.png')
        combined_image.save(triplet_filename)

    return is_correct

def get_image_folders(base_path):
    image_folders = [os.path.join(base_path, folder) for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    blocky_folders = [folder for folder in image_folders if 'blocky' in os.path.basename(folder)]
    smooth_folders = [folder for folder in image_folders if 'smooth' in os.path.basename(folder)]
    return blocky_folders, smooth_folders

def evaluate_trials(object_A_folders, object_B_folders, model, transform, device, base_path, trial, num_trials=100):
    accuracies = []

    for trial_num in tqdm(range(1, num_trials + 1)):
        object_A_folder = random.choice(object_A_folders)
        object_B_folder = random.choice(object_B_folders)

        object_A_images = [os.path.join(object_A_folder, img) for img in os.listdir(object_A_folder)]
        object_B_images = [os.path.join(object_B_folder, img) for img in os.listdir(object_B_folder)]

        if trial_num % 5 == 0:
            is_correct = oddity_detection(object_A_images, object_B_images, model, transform, device, base_path, trial_num=trial_num, trial=trial)
        else:
            is_correct = oddity_detection(object_A_images, object_B_images, model, transform, device, base_path=base_path)

        accuracies.append(is_correct)

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy

if __name__ == "__main__":
    base_path = "/home/vaclav_knapp/shapegen/images_params_3"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    model = load_dino_v2_model().to(device)

    blocky_folders, smooth_folders = get_image_folders(base_path)

    # Smooth vs. Smooth
    accuracy_smooth = evaluate_trials(
        object_A_folders=smooth_folders,
        object_B_folders=smooth_folders,
        model=model,
        transform=transform,
        device=device,
        base_path=base_path,
        num_trials=150,
        trial='smooth_smooth'
    )
    print(f'Smooth vs. Smooth accuracy: {accuracy_smooth:.2f}')

    # Blocky vs. Blocky
    accuracy_blocky = evaluate_trials(
        object_A_folders=blocky_folders,
        object_B_folders=blocky_folders,
        model=model,
        transform=transform,
        device=device,
        base_path=base_path,
        num_trials=150,
        trial='blocky_blocky'
    )
    print(f'Blocky vs. Blocky accuracy: {accuracy_blocky:.2f}')


    with open(os.path.join(base_path, 'plots', 'accuracies.txt'), 'w') as f:
        f.write(f'Smooth vs. Smooth accuracy: {accuracy_smooth:.2f}\n')
        f.write(f'Blocky vs. Blocky accuracy: {accuracy_blocky:.2f}\n')


    accuracies = [accuracy_smooth, accuracy_blocky]
    categories = ['Smooth vs. Smooth', 'Blocky vs. Blocky']

    plt.figure(figsize=(8, 6))
    sns.barplot(x=categories, y=accuracies, palette='viridis')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Oddity Detection Accuracies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'plots', 'accuracy_bar_chart.png'), dpi=300)
    plt.show()
