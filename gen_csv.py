import csv
import os

# Define the path to your CSV file
csv_file_path = "./csv/trainLabels.csv"

# Define the path to the folder containing your files
folder_path = "/Volumes/Data/malware-classification/asm"

# Create a dictionary to store the labels for each file
file_labels = {}

# Open the CSV file and read its contents into the file_labels dictionary
with open(csv_file_path, "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        file_labels[row[0]] = row[1]

# Scan the folder and create a list of all the files in it
files = os.listdir(folder_path)

# Create a new CSV file to store the results
new_csv_file_path = "./csv/trainLabels_new.csv"
with open(new_csv_file_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")

    # Loop through all the files in the folder and check their labels
    for file in files:
        file_name, file_ext = os.path.splitext(file)
        if file_name in file_labels:
            label = file_labels[file_name]
        else:
            label = "Unknown"

        # Write the file name and label to the new CSV file
        writer.writerow([f'"{file_name}"', label])