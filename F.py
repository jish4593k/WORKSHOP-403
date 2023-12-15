from CommandInterface import CommandInterface
from Fragment import Fragment
import torch
import torch.nn.functional as F
from torchvision import transforms
from PyQt5.QtWidgets import QFileDialog
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

class SpeedChangeModel(torch.nn.Module):
    def __init__(self, speed_multiplier):
        super(SpeedChangeModel, self).__init__()
        self.speed_multiplier = speed_multiplier

    def forward(self, x):
        # Apply speed change logic using PyTorch operations
        return F.interpolate(x, scale_factor=self.speed_multiplier, mode='nearest')

class ChangeSpeedCommand(CommandInterface):
    def __str__(self):
        return f"Changing speed of fragment {self.fragment.id + 1} by {self.speed_multiplier} times"

    def __init__(self, time_line, speed_multiplier, fragment: Fragment):
        self.speed_multiplier = speed_multiplier
        self.fragment = fragment
        self.time_line = time_line

    def execute(self):
        # Load the video using OpenCV
        cap = cv2.VideoCapture(self.fragment.clip.filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Read frames and convert to PyTorch tensor
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frames.append(transforms.ToTensor()(frame))

        frames_tensor = torch.stack(frames)

        # Create and load the PyTorch model
        model = SpeedChangeModel(self.speed_multiplier)
        model.eval()

        # Apply the model to the frames
        slowed_down_frames_tensor = model(frames_tensor.unsqueeze(0)).squeeze(0)

        # Convert frames back to numpy array
        slowed_down_frames = [transforms.ToPILImage()(frame) for frame in slowed_down_frames_tensor]
        slowed_down_frames = [transforms.ToTensor()(frame) for frame in slowed_down_frames]

        # Save the new video
        out_filename = f"changed_speed_{self.fragment.id}.mp4"
        out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))

        for frame in slowed_down_frames:
            out.write((frame.numpy() * 255).astype(np.uint8).transpose(1, 2, 0))

        out.release()

        # Set the new clip
        self.time_line.time_line[self.fragment.id].set_clip(out_filename)

        # Show a Seaborn plot for demonstration
        sns.lineplot(x=range(len(frames)), y=[self.speed_multiplier] * len(frames))
        plt.title("Speed Change Over Frames")
        plt.show()

    def undo(self):
        # Reset the clip to the original
        self.time_line.time_line[self.fragment.id].set_clip(self.fragment.clip.filename)

# Example Usage
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])

    # Allow the user to select a video file using a PyQt file dialog
    file_dialog = QFileDialog()
    file_dialog.setNameFilter("Video files (*.mp4 *.avi)")
    file_dialog.setDefaultSuffix("mp4")
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    if file_dialog.exec_():
        video_file = file_dialog.selectedFiles()[0]
        fragment = Fragment(video_file)

        # Create a ChangeSpeedCommand and execute it
        change_speed_command = ChangeSpeedCommand(time_line=None, speed_multiplier=0.5, fragment=fragment)
        change_speed_command.execute()

    app.exec_()
