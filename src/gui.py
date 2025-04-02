#!/usr/bin/env python3
#
#  gui.py
#
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#


import tkinter as tk
from tkinter import ttk
import numpy as np
from tkinter import scrolledtext
import subprocess
import os
import re  # Import the regular expression module

def actions_tab(tab_control):
    actions_frame = ttk.Frame(tab_control)
    tab_control.add(actions_frame, text="Actions")

    tk.Label(actions_frame, text="Select Action:").pack(pady=10)

    action_var = tk.StringVar()
    action_var.set("train")  # Default action

    train_radio = tk.Radiobutton(actions_frame, text="Train", variable=action_var, value="train")
    train_radio.pack(anchor=tk.W)

    test_radio = tk.Radiobutton(actions_frame, text="Test", variable=action_var, value="test")
    test_radio.pack(anchor=tk.W)

    tk.Label(actions_frame, text="Select Model:").pack(pady=10)

    model_var = tk.StringVar()
    model_var.set("nn")  # Default model

    nn_radio = tk.Radiobutton(actions_frame, text="NN", variable=model_var, value="nn")
    nn_radio.pack(anchor=tk.W)

    svr_radio = tk.Radiobutton(actions_frame, text="SVR", variable=model_var, value="svr")
    svr_radio.pack(anchor=tk.W)

    result_text = scrolledtext.ScrolledText(actions_frame, height=10, width=50)
    result_text.pack(pady=10)
    

    def perform_action():
        action = action_var.get()
        model = model_var.get()

        try:
            current_dir = os.getcwd()
            main_script_path = os.path.join(current_dir, "main.py")

            if not os.path.exists(main_script_path):
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Error: main.py not found in {current_dir}\n")
                return

            command = ["python", main_script_path, "-a", action, "-m", model]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=current_dir
            )

            result_text.delete(1.0, tk.END)  # Clear the text widget before inserting new text.

            for line in process.stdout:
                result_text.insert(tk.END, line)
                result_text.see(tk.END)

                # Search for the specified pattern
                match = re.search(r"Predicted Weights \(real, imaginary\): \((-?\d+\.?\d*e?-?\d*),(-?\d+\.?\d*e?-?\d*)\)", line)

                if match:
                    result_text.delete(1.0, tk.END)  # Clear the text widget before inserting new text.
                    result_text.insert(tk.END, f"{match.group(0)}\n")  # Print the matched line

            return_code = process.poll()
            if return_code != 0:
                result_text.insert(tk.END, f"Error: main.py exited with code {return_code}\n")
            else:
                result_text.insert(tk.END, "main.py executed successfully.\n")

        except FileNotFoundError:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "Error: Python executable not found.\n")
        except Exception as e:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

    action_button = tk.Button(actions_frame, text="Perform Action", command=perform_action)
    action_button.pack(pady=10)

def insights_tab(tab_control):
    insights_frame = ttk.Frame(tab_control)
    tab_control.add(insights_frame, text="Insights")

    tk.Label(insights_frame, text="Insights Display:").pack(pady=10)

    insights_text = scrolledtext.ScrolledText(insights_frame, height=15, width=60)
    insights_text.pack(pady=5)

    def display_insights():
        insights_text.delete(1.0, tk.END)  # Clear previous text

        try:
            # Assuming data_insights.py is in the current directory
            current_dir = os.getcwd()
            insights_script_path = os.path.join(current_dir, "data_insights.py")

            if not os.path.exists(insights_script_path):
                insights_text.insert(tk.END, f"Error: data_insights.py not found in {current_dir}\n")
                return

            process = subprocess.Popen(
                ["python", insights_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=current_dir #set working directory to current directory.
            )

            output, _ = process.communicate() #Get the entire output

            return_code = process.poll()
            if return_code != 0:
                insights_text.insert(tk.END, f"Error: data_insights.py exited with code {return_code}\n")
            else:
                insights_text.insert(tk.END, output) #insert entire output.

        except FileNotFoundError:
            insights_text.insert(tk.END, "Error: Python executable not found.\n")
        except Exception as e:
            insights_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

    insights_button = tk.Button(insights_frame, text="Generate Insights", command=display_insights)
    insights_button.pack(pady=5)

def generate_tab(tab_control):
    generate_frame = ttk.Frame(tab_control)
    tab_control.add(generate_frame, text="Generate")

    tk.Label(generate_frame, text="Data Generation Log:").pack(pady=10)

    log_text = scrolledtext.ScrolledText(generate_frame, height=15, width=60)
    log_text.pack(pady=5)

    def generate_data():
        log_text.delete(1.0, tk.END)  # Clear previous log

        try:
            # Assuming generate_data.py is in the 'src' directory
            src_dir = os.getcwd()
            generate_script_path = os.path.join(src_dir, "generate_data.py")

            if not os.path.exists(generate_script_path):
                log_text.insert(tk.END, f"Error: generate_data.py not found in {src_dir}\n")
                return

            process = subprocess.Popen(
                ["python", generate_script_path, "-g"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=src_dir #set working directory to src.
            )

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    log_text.insert(tk.END, output)
                    log_text.see(tk.END)  # Scroll to the bottom

            return_code = process.poll()
            if return_code != 0:
                log_text.insert(tk.END, f"Error: generate_data.py exited with code {return_code}\n")
            else:
                log_text.insert(tk.END, "Data generation completed.\n")
        except FileNotFoundError:
            log_text.insert(tk.END, "Error: Python executable not found.\n")
        except Exception as e:
            log_text.insert(tk.END, f"An unexpected error occurred: {e}\n")

    generate_button = tk.Button(generate_frame, text="Generate Data", command=generate_data) #Corrected line
    generate_button.pack(pady=5)


root = tk.Tk()
root.title("Deep MIMO AI")

tab_control = ttk.Notebook(root)

actions_tab(tab_control)
insights_tab(tab_control)
generate_tab(tab_control)

tab_control.pack(expand=1, fill="both")

root.mainloop()
