# import re

# # Path to the train_model.py file
# file_path = 'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\train_model.py'

# # The new loss function you want to set
# new_loss_function = 'loss_function = LabelSmoothingCrossEntropy(smoothing=smoothing)'

# # Read the contents of the train_model.py file
# with open(file_path, 'r') as file:
#     content = file.read()

# # Find the line where loss_function is assigned
# # Replace it with the new loss function
# content = re.sub(r'loss_function = nn.CrossEntropyLoss\(\)', new_loss_function, content)

# # Write the updated content back to the file
# with open(file_path, 'w') as file:
#     file.write(content)

# print(f"Updated {file_path} to use LabelSmoothingCrossEntropy.")



# import re

# # Path to the modeling.py file
# file_path = 'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\utils\\modeling.py'

# # Read the contents of the modeling.py file
# with open(file_path, 'r') as file:
#     content = file.read()

# new_init = '''self.linear = nn.Linear(self.bert.config.hidden_size*4, self.bert.config.hidden_size) #1'''
# new_forward = '''cat = torch.cat((txt_mean, topic_mean, txt_mean - topic_mean, txt_mean * topic_mean), dim=1)'''

# # Update the __init__ method
# content = re.sub(r'self\.linear = nn\.Linear\(self\.bart\.config.hidden_size\*2, self\.bart\.config\.hidden_size\) #1', new_init, content)

# # Update the forward method
# content = re.sub(r'cat = torch\.cat\(\(txt_mean, topic_mean\), dim=1\) #1', new_forward, content)

# # Write the updated content back to the file
# with open(file_path, 'w') as file:
#     file.write(content)





import shutil
import os

# List of file paths to move
file_paths = [
    'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\best_loss_results_test_df.csv',  # Replace with actual file paths
    'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\best_loss_results_validation_df.csv',
    'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\best_results_test_df.csv',
    'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\best_results_validation_df.csv',
    'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\results_training_df.csv',
    'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\results_test_df.csv',
    'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\results_validation_df.csv',
    'C:\\Users\\CSE RGUKT\\Downloads\\TTS\\TTS\\TTS_zeroshot\\src\\checkpoint.pt'
]

# Destination directory where you want to move the files
destination_dir = r"D:\Bart\100 percent\multiple seeds\normal atten, labels smotting"

# Ensure the destination folder exists, if not, create it
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Loop through the files and move them to the destination folder
for file_path in file_paths:
    if os.path.exists(file_path):
        # Move file to destination folder
        shutil.move(file_path, destination_dir)
        # print(f"Moved {file_path} to {destination_dir}")

# print("All files have been moved.")
