import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
from predict import load_model, predict_price

class VehicleAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Price Prediction")
        self.root.geometry("800x600")
        
        # Load model and artifacts
        self.model, self.scaler, self.feature_info = load_model()
        if self.model is None or self.scaler is None or self.feature_info is None:
            messagebox.showerror("Error", "Failed to load model. Please check model files.")
            sys.exit(1)
        
        # Style configuration
        self.setup_styles()
        
        # Load available options
        self.load_options()
        
        # Create main notebook
        self.create_notebook()
        
        # Setup prediction tab
        self.setup_predict_tab()

    def setup_styles(self):
        style = ttk.Style()
        style.configure('Custom.TFrame', background='#f0f0f0')
        style.configure('Custom.TLabel', font=('Helvetica', 12))
        style.configure('Custom.TButton', font=('Helvetica', 12), padding=5)
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Result.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Custom.TCombobox', font=('Helvetica', 12))

    def load_options(self):
        # Load the data to get available options
        try:
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'vehicles.csv')
            df = pd.read_csv(data_path)
            
            # Get unique values for each category, handling mixed data types
            self.options = {
                'year': sorted(df['year'].dropna().astype(int).unique()),
                'manufacturer': sorted(df['manufacturer'].dropna().astype(str).str.lower().unique()),
                'model': sorted(df['model'].dropna().astype(str).str.lower().unique()),
                'condition': sorted(df['condition'].dropna().astype(str).str.lower().unique()),
                'fuel': sorted(df['fuel'].dropna().astype(str).str.lower().unique()),
                'title_status': sorted(df['title_status'].dropna().astype(str).str.lower().unique()),
                'transmission': sorted(df['transmission'].dropna().astype(str).str.lower().unique()),
                'drive': sorted(df['drive'].dropna().astype(str).str.lower().unique()),
                'type': sorted(df['type'].dropna().astype(str).str.lower().unique()),
                'paint_color': sorted(df['paint_color'].dropna().astype(str).str.lower().unique())
            }
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load options: {str(e)}")
            # Fallback options
            current_year = pd.Timestamp.now().year
            self.options = {
                'year': list(range(1990, current_year + 1)),
                'manufacturer': ['toyota', 'honda', 'ford', 'chevrolet', 'bmw'],
                'model': ['camry', 'accord', 'f-150', 'silverado', '3 series'],
                'condition': ['excellent', 'good', 'fair', 'like new', 'new'],
                'fuel': ['gas', 'diesel', 'hybrid', 'electric', 'other'],
                'title_status': ['clean', 'salvage', 'rebuilt', 'parts only'],
                'transmission': ['automatic', 'manual', 'other'],
                'drive': ['4wd', 'fwd', 'rwd'],
                'type': ['sedan', 'truck', 'suv', 'coupe', 'hatchback'],
                'paint_color': ['black', 'white', 'silver', 'red', 'blue']
            }

    def create_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Create prediction tab
        self.predict_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_frame, text="Price Prediction")

    def setup_predict_tab(self):
        # Create input frame
        input_frame = ttk.Frame(self.predict_frame, style='Custom.TFrame')
        input_frame.pack(padx=20, pady=20, fill='x')
        
        # Create variables for input fields
        self.year_var = tk.StringVar()
        self.manufacturer_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.condition_var = tk.StringVar()
        self.odometer_var = tk.StringVar()
        self.fuel_var = tk.StringVar()
        self.title_status_var = tk.StringVar()
        self.transmission_var = tk.StringVar()
        self.drive_var = tk.StringVar()
        self.type_var = tk.StringVar()
        self.paint_color_var = tk.StringVar()
        
        # Create input fields with dropdowns
        ttk.Label(input_frame, text="Year:", style='Custom.TLabel').grid(row=0, column=0, padx=5, pady=5, sticky='w')
        year_combo = ttk.Combobox(input_frame, textvariable=self.year_var, 
                                 values=self.options['year'], style='Custom.TCombobox')
        year_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Manufacturer:", style='Custom.TLabel').grid(row=1, column=0, padx=5, pady=5, sticky='w')
        manufacturer_combo = ttk.Combobox(input_frame, textvariable=self.manufacturer_var, 
                                        values=self.options['manufacturer'], style='Custom.TCombobox')
        manufacturer_combo.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Model:", style='Custom.TLabel').grid(row=2, column=0, padx=5, pady=5, sticky='w')
        model_combo = ttk.Combobox(input_frame, textvariable=self.model_var, 
                                  values=self.options['model'], style='Custom.TCombobox')
        model_combo.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Condition:", style='Custom.TLabel').grid(row=3, column=0, padx=5, pady=5, sticky='w')
        condition_combo = ttk.Combobox(input_frame, textvariable=self.condition_var, 
                                      values=self.options['condition'], style='Custom.TCombobox')
        condition_combo.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Odometer:", style='Custom.TLabel').grid(row=4, column=0, padx=5, pady=5, sticky='w')
        odometer_entry = ttk.Entry(input_frame, textvariable=self.odometer_var)
        odometer_entry.grid(row=4, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Fuel Type:", style='Custom.TLabel').grid(row=5, column=0, padx=5, pady=5, sticky='w')
        fuel_combo = ttk.Combobox(input_frame, textvariable=self.fuel_var, 
                                 values=self.options['fuel'], style='Custom.TCombobox')
        fuel_combo.grid(row=5, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Title Status:", style='Custom.TLabel').grid(row=6, column=0, padx=5, pady=5, sticky='w')
        title_status_combo = ttk.Combobox(input_frame, textvariable=self.title_status_var, 
                                         values=self.options['title_status'], style='Custom.TCombobox')
        title_status_combo.grid(row=6, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Transmission:", style='Custom.TLabel').grid(row=7, column=0, padx=5, pady=5, sticky='w')
        transmission_combo = ttk.Combobox(input_frame, textvariable=self.transmission_var, 
                                         values=self.options['transmission'], style='Custom.TCombobox')
        transmission_combo.grid(row=7, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Drive Type:", style='Custom.TLabel').grid(row=8, column=0, padx=5, pady=5, sticky='w')
        drive_combo = ttk.Combobox(input_frame, textvariable=self.drive_var, 
                                  values=self.options['drive'], style='Custom.TCombobox')
        drive_combo.grid(row=8, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Vehicle Type:", style='Custom.TLabel').grid(row=9, column=0, padx=5, pady=5, sticky='w')
        type_combo = ttk.Combobox(input_frame, textvariable=self.type_var, 
                                 values=self.options['type'], style='Custom.TCombobox')
        type_combo.grid(row=9, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Paint Color:", style='Custom.TLabel').grid(row=10, column=0, padx=5, pady=5, sticky='w')
        paint_color_combo = ttk.Combobox(input_frame, textvariable=self.paint_color_var, 
                                        values=self.options['paint_color'], style='Custom.TCombobox')
        paint_color_combo.grid(row=10, column=1, padx=5, pady=5)
        
        # Create predict button
        ttk.Button(input_frame, text="Predict Price", command=self.predict_price, 
                  style='Custom.TButton').grid(row=11, column=0, columnspan=2, pady=20)
        
        # Create result label
        self.result_label = ttk.Label(input_frame, text="", style='Result.TLabel')
        self.result_label.grid(row=12, column=0, columnspan=2, pady=10)

    def predict_price(self):
        try:
            # Validate inputs
            try:
                year = int(self.year_var.get())
                if not (1900 <= year <= 2024):
                    raise ValueError("Year must be between 1900 and 2024")
            except ValueError:
                messagebox.showerror("Error", "Please select a valid year")
                return

            try:
                odometer = float(self.odometer_var.get().replace(',', ''))
                if not (0 <= odometer <= 500000):
                    raise ValueError("Odometer must be between 0 and 500,000")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid odometer reading between 0 and 500,000")
                return

            # Create input data dictionary
            input_data = {
                'year': year,
                'manufacturer': self.manufacturer_var.get().lower(),
                'model': self.model_var.get().lower(),
                'condition': self.condition_var.get().lower(),
                'odometer': odometer,
                'fuel': self.fuel_var.get().lower(),
                'title_status': self.title_status_var.get().lower(),
                'transmission': self.transmission_var.get().lower(),
                'drive': self.drive_var.get().lower(),
                'type': self.type_var.get().lower(),
                'paint_color': self.paint_color_var.get().lower()
            }

            # Make prediction using already loaded model and artifacts
            prediction = predict_price(input_data, self.model, self.scaler, self.feature_info)
            
            if prediction is not None:
                self.result_label.config(text=f"Predicted Price: ${prediction:,.2f}")
            else:
                messagebox.showerror("Error", "Failed to make prediction. Please try again.")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = VehicleAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 