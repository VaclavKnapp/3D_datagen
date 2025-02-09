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
import argparse

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

def oddity_detection(object_A, object_B, model, transform, device, base_path, trial_num=None, factory_name=None, n_objects=None):
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


    if trial_num is not None and factory_name is not None and n_objects is not None:

        plot_dir = os.path.join(base_path, 'plots', n_objects, factory_name)
        os.makedirs(plot_dir, exist_ok=True)
        labels = ['Image 1', 'Image 2', 'Image 3']

        plt.figure(figsize=(6, 5))
        sns.heatmap(similarity_matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap='viridis')
        plt.title(f'Similarity Matrix - {factory_name} Trial {trial_num}')
        plt.tight_layout()
        heatmap_filename = os.path.join(plot_dir, f'similarity_heatmap_{factory_name}_trial_{trial_num}.png')
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
        text_width, text_height = draw.textbbox((0, 0), text_correct, font=font)[2:]
        x_text = x_positions[b_index] + images[b_index].width / 2 - text_width / 2
        y_text = y0 - text_height - 5
        if y_text < 0:
            y_text = y1 + 5
        draw.text((x_text, y_text), text_correct, fill=(0, 255, 0), font=font)


        if b_index_guess != b_index:
            text_guess = 'Model Guess'
            fill_color = (255, 0, 0)
        else:
            text_guess = 'Model Correct'
            fill_color = (0, 255, 0)
        text_width, text_height = draw.textbbox((0, 0), text_correct, font=font)[2:]
        x_text = x_positions[b_index_guess] + images[b_index_guess].width / 2 - text_width / 2
        y_text = y0 - text_height - 5
        if y_text < 0:
            y_text = y1 + 5
        draw.text((x_text, y_text), text_guess, fill=fill_color, font=font)


        triplet_filename = os.path.join(plot_dir, f'triplet_{factory_name}_trial_{trial_num}.png')
        combined_image.save(triplet_filename)

    return is_correct

def get_all_image_folders(base_path):

    n_objects_dirs = [os.path.join(base_path, dir_name) for dir_name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, dir_name))]

    factory_image_folders = {}
    for n_objects_dir in n_objects_dirs:
        n_objects = os.path.basename(n_objects_dir)

        example_dirs = [os.path.join(n_objects_dir, dir_name) for dir_name in os.listdir(n_objects_dir) if os.path.isdir(os.path.join(n_objects_dir, dir_name))]

        for example_dir in example_dirs:

            object_dirs = [os.path.join(example_dir, dir_name) for dir_name in os.listdir(example_dir) if os.path.isdir(os.path.join(example_dir, dir_name))]


            factory_image_folders[example_dir] = {'n_objects': n_objects, 'image_folders': object_dirs}

    return factory_image_folders

def evaluate_factory(image_folders, model, transform, device, base_path, num_trials=100, factory_name=None, n_objects=None):
    accuracies = []

    for trial_num in range(1, num_trials + 1):
        object_A_folder, object_B_folder = random.sample(image_folders, 2)

        object_A_images = [os.path.join(object_A_folder, img) for img in os.listdir(object_A_folder)]
        object_B_images = [os.path.join(object_B_folder, img) for img in os.listdir(object_B_folder)]

        if trial_num % 10 == 0:
            is_correct = oddity_detection(object_A_images, object_B_images, model, transform, device, trial_num=trial_num, factory_name=factory_name, n_objects=n_objects, base_path=base_path)
        else:
            is_correct = oddity_detection(object_A_images, object_B_images, model, transform, device, factory_name=factory_name, n_objects=n_objects, base_path=base_path)

        accuracies.append(is_correct)

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy

def evaluate_all_factories(base_path, model, transform, device, num_trials=100, zeroverse=False):
    factory_image_folders = get_all_image_folders(base_path)
    accuracies = {}

    for factory, data in tqdm(factory_image_folders.items()):
        image_folders = data['image_folders']
        n_objects = data['n_objects']
        if len(image_folders) > 1:
            factory_name = os.path.basename(factory)
            accuracy = evaluate_factory(image_folders, model, transform, device, base_path, num_trials, factory_name=factory_name, n_objects=n_objects)

            if n_objects not in accuracies:
                accuracies[n_objects] = {}
            accuracies[n_objects][factory_name] = accuracy


    for n_objects, factory_accuracies in accuracies.items():

        plot_dir = os.path.join(base_path, 'plots', n_objects)
        os.makedirs(plot_dir, exist_ok=True)
        txt_file = os.path.join(plot_dir, 'accuracy.txt')


        with open(txt_file, 'w') as file:
            for factory_name, accuracy in factory_accuracies.items():
                print(f'{factory_name}: {accuracy:.2f}')
                file.write(f'{factory_name}: {accuracy:.2f}\n')

        if zeroverse:

            pass
        else:

            sorted_accuracies = sorted(factory_accuracies.items(), key=lambda x: x[1], reverse=True)
            factories = [item[0] for item in sorted_accuracies]
            accuracy_values = [item[1] for item in sorted_accuracies]

            plt.figure(figsize=(12, 6))
            plt.bar(factories, accuracy_values, color='skyblue')
            plt.xlabel('Factory')
            plt.ylabel('Accuracy')
            plt.title(f'Oddity Detection Accuracy per Factory (n_objects={n_objects})')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'accuracy_per_factory.png'), dpi=300)
            plt.close()


    mean_accuracies = {}
    for n_objects, factory_accuracies in accuracies.items():
        mean_accuracy = np.mean(list(factory_accuracies.values()))
        mean_accuracies[int(n_objects)] = mean_accuracy


    sorted_n_objects = sorted(mean_accuracies.keys())
    mean_accuracy_values = [mean_accuracies[n] for n in sorted_n_objects]


    plt.figure(figsize=(8, 6))
    plt.plot(sorted_n_objects, mean_accuracy_values, marker='o')
    plt.xlabel('Number of Objects (n_objects)')
    plt.ylabel('Mean Accuracy')
    plt.title('Mean Accuracy vs Number of Objects')
    plt.grid(True)
    plt.tight_layout()
    overall_plot_dir = os.path.join(base_path, 'plots')
    os.makedirs(overall_plot_dir, exist_ok=True)
    plt.savefig(os.path.join(overall_plot_dir, 'mean_accuracy_vs_n_objects.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Oddity Detection Evaluation')
    parser.add_argument('--zeroverse', action='store_true', help='Create graph of number of shapes vs oddity accuracy')

    args = parser.parse_args()

    base_path = "/home/vaclav_knapp/zeroverse/berkeley_images_params_4"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_dino_v2_model().to(device)

    evaluate_all_factories(base_path, model, transform, device, num_trials=150, zeroverse=args.zeroverse)

