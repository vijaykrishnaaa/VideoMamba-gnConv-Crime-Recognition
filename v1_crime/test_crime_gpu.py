import os, sys, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser('~/vlil_40/VideoMamba/mamba'))
sys.path.insert(0, os.path.expanduser('~/vlil_40/VideoMamba/videomamba/video_sm'))
from models.videomamba import videomamba_small

DATA_ROOT = os.path.expanduser('~/final_year_project_29/dataset_new_split')
TEST_DIR  = os.path.join(DATA_ROOT, 'test')
MODEL_PATH = os.path.expanduser('~/vlil_40/checkpoints_crime/best_crime_v6.pth')
NUM_FRAMES = 8

class CrimeDataset(Dataset):
    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.MOV'}
    def __init__(self, split_dir, class_to_idx):
        self.video_paths, self.labels = [], []
        for c in sorted(os.listdir(split_dir)):
            d = os.path.join(split_dir, c)
            if not os.path.isdir(d): continue
            for f in sorted(os.listdir(d)):
                if os.path.splitext(f)[1].lower() in self.VIDEO_EXTS:
                    self.video_paths.append(os.path.join(d, f))
                    self.labels.append(class_to_idx[c])
        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.video_paths)
    def __getitem__(self, idx):
        try:
            import cv2
            cv2.setNumThreads(0)
            cap = cv2.VideoCapture(self.video_paths[idx])
            frames_list = []
            for _ in range(300):
                ret, f = cap.read()
                if not ret: break
                frames_list.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cap.release()
            total = len(frames_list)
            if total == 0: raise ValueError
            idx_seq = np.pad(np.arange(total), (0, max(0, NUM_FRAMES - total)), mode='wrap') if total < NUM_FRAMES else np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
            x = torch.stack([self.transform(frames_list[int(i)]) for i in idx_seq]).permute(1, 0, 2, 3)
            return x, self.labels[idx]
        except:
            return torch.zeros(3, NUM_FRAMES, 224, 224), self.labels[idx]

def main():
    c_folders = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
    c2i = {n: i for i, n in enumerate(c_folders)}
    loader = DataLoader(CrimeDataset(TEST_DIR, c2i), batch_size=2, shuffle=False, num_workers=0)
    
    model = videomamba_small(pretrained=False, num_classes=len(c2i))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model = model.cuda()  # <--- Moved to GPU!
    model.eval()

    corr, tot = 0, 0
    pc_corr, pc_tot = [0]*len(c2i), [0]*len(c2i)
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            preds = model(x).argmax(1)
            corr += (preds == y).sum().item()
            tot += x.size(0)
            for p, l in zip(preds.cpu().numpy(), y.cpu().numpy()):
                pc_tot[l] += 1
                if p == l: pc_corr[l] += 1

    print("\n" + "="*50)
    print(f"FINAL TEST ACCURACY: {100 * corr / max(tot, 1):.2f}%")
    print("="*50)
    for i in range(len(c2i)):
        if pc_tot[i] > 0: print(f"  {c_folders[i]:15s}: {pc_corr[i]:2d}/{pc_tot[i]:2d} ({100*pc_corr[i]/pc_tot[i]:.1f}%)")

if __name__ == '__main__': main()
