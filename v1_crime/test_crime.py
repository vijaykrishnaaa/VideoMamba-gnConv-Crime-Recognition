import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.expanduser('~/vlil_40/VideoMamba/mamba'))
sys.path.insert(0, os.path.expanduser('~/vlil_40/VideoMamba/videomamba/video_sm'))
from models.videomamba import videomamba_small

# Config
DATA_ROOT = os.path.expanduser('~/final_year_project_29/dataset_new_split')
TEST_DIR  = os.path.join(DATA_ROOT, 'test')
MODEL_PATH = os.path.expanduser('~/vlil_40/checkpoints_crime/best_crime_v6.pth')
NUM_FRAMES = 8

class CrimeDataset(Dataset):
    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.MOV'}
    def __init__(self, split_dir, class_to_idx):
        self.video_paths = []
        self.labels = []
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir): continue
            label = class_to_idx[class_name]
            for fname in sorted(os.listdir(class_dir)):
                if os.path.splitext(fname)[1].lower() in self.VIDEO_EXTS:
                    self.video_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.video_paths)

    def __getitem__(self, idx):
        path, label = self.video_paths[idx], self.labels[idx]
        try:
            import cv2
            cv2.setNumThreads(0)
            cap = cv2.VideoCapture(path)
            frames_list = []
            for _ in range(300):
                ret, frame = cap.read()
                if not ret: break
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            total = len(frames_list)
            if total == 0:
                raise ValueError("No frames")
            indices = np.pad(np.arange(total), (0, max(0, NUM_FRAMES - total)), mode='wrap') if total < NUM_FRAMES else np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
            frames = [frames_list[int(i)] for i in indices]
            x = torch.stack([self.transform(f) for f in frames]).permute(1, 0, 2, 3)
            return x, label
        except Exception as e:
            return torch.zeros(3, NUM_FRAMES, 224, 224), label

def main():
    print("Loading test dataset...")
    class_folders = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
    class_to_idx = {name: i for i, name in enumerate(class_folders)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    
    test_dataset = CrimeDataset(TEST_DIR, class_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"Loaded {len(test_dataset)} test videos.")

    print("Building model & loading weights...")
    model = videomamba_small(pretrained=False, num_classes=len(class_to_idx))
    
    # Load your saved weights!
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model = model.cpu()
    model.eval()

    print("Running evaluation (this may take a few minutes on CPU)...")
    correct, total = 0, 0
    per_class_correct = [0] * len(class_to_idx)
    per_class_total = [0] * len(class_to_idx)

    with torch.no_grad():
        for x, y in tqdm(test_loader):
            logits = model(x)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            for p, l in zip(preds.numpy(), y.numpy()):
                per_class_total[l] += 1
                if p == l: per_class_correct[l] += 1

    print("\n" + "="*50)
    print(f"FINAL TEST ACCURACY: {100 * correct / total:.2f}%")
    print("="*50)
    for i in range(len(class_to_idx)):
        if per_class_total[i] > 0:
            print(f"  {idx_to_class[i]:15s}: {per_class_correct[i]:2d}/{per_class_total[i]:2d} ({100*per_class_correct[i]/per_class_total[i]:.1f}%)")

if __name__ == '__main__':
    main()
