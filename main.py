import tkinter as tk

from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import tempfile
import os
import time
from sklearn.decomposition import PCA

class GeologicalVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Geological Data Visualizer")
        self.root.geometry("700x800")
        self.root.configure(bg='#eceff1')

        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=('Helvetica', 12, 'bold'), padding=10, background='#4CAF50', foreground='white')
        style.map("TButton", background=[('active', '#45a049')], foreground=[('active', 'white')])
        style.configure("TLabel", font=('Helvetica', 12, 'bold'), background='#eceff1', foreground='#212121')
        style.configure("TFrame", background='#eceff1')
        style.configure("TLabelframe", background='#eceff1', foreground='#212121', padding=15)
        style.configure("TLabelframe.Label", font=('Helvetica', 14, 'bold'), foreground='#212121')

        self.data = None
        self.main_frame = ttk.Frame(root, padding="20", relief="raised", borderwidth=2)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.create_control_panel()

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")])
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    self.data = pd.read_excel(file_path)
                if len(self.data) > 5000:  # Reduced sample size
                    self.data = self.data.sample(n=5000, random_state=42)
                self.update_combos()
                messagebox.showinfo("Success", "File loaded successfully!")
                self.visualize_button.configure(state='normal')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def update_combos(self):
        if self.data is not None:
            columns = list(self.data.columns)
            for combo in [self.x_combo, self.y_combo, self.z_combo, self.color_combo]:
                combo['values'] = columns
                combo.configure(state='readonly')
            if columns:
                self.x_var.set('Ore Pulp Density' if 'Ore Pulp Density' in columns else columns[0])
                self.y_var.set('% Iron Feed' if '% Iron Feed' in columns else columns[1] if len(columns) > 1 else columns[0])
                self.z_var.set('% Iron Feed' if '% Iron Feed' in columns else columns[2] if len(columns) > 2 else columns[0])
                self.color_var.set('Ore Pulp Flow' if 'Ore Pulp Flow' in columns else columns[3] if len(columns) > 3 else columns[0])

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.main_frame, text="Control Panel", padding="15")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ttk.Button(control_frame, text="Load File", command=self.load_file).grid(row=0, column=0, columnspan=2, pady=(0, 25), padx=10)

        ttk.Label(control_frame, text="X-Axis:").grid(row=1, column=0, padx=10, pady=15, sticky="e")
        self.x_var = tk.StringVar(value='')
        self.x_combo = ttk.Combobox(control_frame, textvariable=self.x_var, width=25, state='disabled')
        self.x_combo.grid(row=1, column=1, padx=10, pady=15, sticky="w")

        ttk.Label(control_frame, text="Y-Axis:").grid(row=2, column=0, padx=10, pady=15, sticky="e")
        self.y_var = tk.StringVar(value='')
        self.y_combo = ttk.Combobox(control_frame, textvariable=self.y_var, width=25, state='disabled')
        self.y_combo.grid(row=2, column=1, padx=10, pady=15, sticky="w")

        ttk.Label(control_frame, text="Z-Axis:").grid(row=3, column=0, padx=10, pady=15, sticky="e")
        self.z_var = tk.StringVar(value='')
        self.z_combo = ttk.Combobox(control_frame, textvariable=self.z_var, width=25, state='disabled')
        self.z_combo.grid(row=3, column=1, padx=10, pady=15, sticky="w")

        ttk.Label(control_frame, text="Color:").grid(row=4, column=0, padx=10, pady=15, sticky="e")
        self.color_var = tk.StringVar(value='')
        self.color_combo = ttk.Combobox(control_frame, textvariable=self.color_var, width=25, state='disabled')
        self.color_combo.grid(row=4, column=1, padx=10, pady=15, sticky="w")

        ttk.Label(control_frame, text="Colormap:").grid(row=5, column=0, padx=10, pady=15, sticky="e")
        self.cmap_var = tk.StringVar(value='Viridis')
        ttk.Combobox(control_frame, textvariable=self.cmap_var,
                    values=['Viridis', 'Plasma', 'Inferno', 'Magma', 'Coolwarm', 'RdBu'], width=25).grid(row=5, column=1, padx=10, pady=15, sticky="w")

        ttk.Label(control_frame, text="Transparency:").grid(row=6, column=0, padx=10, pady=15, sticky="e")
        self.alpha_var = tk.DoubleVar(value=0.7)
        ttk.Scale(control_frame, from_=0.1, to=1.0, variable=self.alpha_var, length=150).grid(row=6, column=1, padx=10, pady=15, sticky="w")

        self.visualize_button = ttk.Button(control_frame, text="Visualize Graphs", command=self.update_plots, state='disabled')
        self.visualize_button.grid(row=7, column=0, columnspan=2, pady=(25, 0), padx=10)

        for i in range(8):
            control_frame.grid_rowconfigure(i, weight=1)
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=2)

    def check_data_variation(self, x_data, y_data):
        x_unique = len(np.unique(x_data))
        y_unique = len(np.unique(y_data))
        return x_unique > 1 and y_unique > 1

    def update_plots(self):
        if self.data is None:
            messagebox.showwarning("No Data", "Please load a file first!")
            return

        loading_label = ttk.Label(self.main_frame, text="Processing... Please wait.", font=('Helvetica', 12))
        loading_label.grid(row=1, column=0, pady=10)
        self.root.update()

        x_data = self.data[self.x_var.get()]
        y_data = self.data[self.y_var.get()]
        z_data = self.data[self.z_var.get()]
        color_data = self.data[self.color_var.get()]
        cmap = self.cmap_var.get()
        alpha = self.alpha_var.get()

        # Normalize color data
        color_data_normalized = (color_data - color_data.min()) / (color_data.max() - color_data.min())

        try:
            # Tab 1: Correlation Matrix, Scatter Plot, 3D Surface for % Iron Feed
            fig1 = make_subplots(rows=3, cols=1,
                               subplot_titles=('Correlation Matrix', '% Iron Feed vs % Iron Concentrate', '3D Surface: % Iron Feed'),
                               specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "scene"}]],
                               vertical_spacing=0.1)

            # Plot 1: Correlation Matrix (limited to key columns)
            relevant_cols = ['% Iron Feed', '% Iron Concentrate', 'Ore Pulp Flow', 'Starch Flow']
            corr_data = self.data[relevant_cols].corr() if all(col in self.data.columns for col in relevant_cols) else self.data.corr()
            fig1.add_trace(
                go.Heatmap(z=corr_data.values, x=corr_data.index, y=corr_data.columns,
                          colorscale=cmap, zmin=-1, zmax=1, showscale=True,
                          colorbar=dict(title="Correlation", len=0.5, thickness=10, tickfont=dict(size=10)),
                          hoverinfo='z'),
                row=1, col=1
            )

            # Plot 2: Scatter Plot
            fig1.add_trace(
                go.Scatter(x=self.data['% Iron Feed'] if '% Iron Feed' in self.data.columns else x_data,
                          y=self.data['% Iron Concentrate'] if '% Iron Concentrate' in self.data.columns else y_data,
                          mode='markers', marker=dict(color=color_data_normalized, colorscale=cmap, opacity=alpha,
                                                     size=6, line=dict(width=0.5, color='DarkSlateGrey'),
                                                     colorbar=dict(title=self.color_var.get(), len=0.5, thickness=10)),
                          hovertemplate='X: %{x}<br>Y: %{y}<br>Color: %{marker.color}<extra></extra>'),
                row=2, col=1
            )

            # Plot 3: 3D Surface
            if self.check_data_variation(x_data, y_data):
                x_unique = np.linspace(min(x_data), max(x_data), 10)
                y_unique = np.linspace(min(y_data), max(y_data), 10)
                X_grid, Y_grid = np.meshgrid(x_unique, y_unique)
                try:
                    Z_grid = griddata((x_data, y_data), z_data, (X_grid, Y_grid), method='cubic')
                    color_grid = griddata((x_data, y_data), color_data, (X_grid, Y_grid), method='cubic')
                except:
                    Z_grid = griddata((x_data, y_data), z_data, (X_grid, Y_grid), method='nearest')
                    color_grid = griddata((x_data, y_data), color_data, (X_grid, Y_grid), method='nearest')
                color_grid_normalized = (color_grid - color_grid.min()) / (color_grid.max() - color_grid.min())
                fig1.add_trace(
                    go.Surface(x=X_grid, y=Y_grid, z=Z_grid, surfacecolor=color_grid_normalized, colorscale=cmap, opacity=alpha,
                              showscale=True, colorbar=dict(title=self.color_var.get(), len=0.5, thickness=10),
                              contours=dict(z=dict(show=True, color='white', width=1)),  # Changed width to 1
                              lighting=dict(ambient=0.5, diffuse=0.8, specular=0.1)),
                    row=3, col=1
                )

            fig1.update_layout(title_text="Geological Data Analysis", title_font=dict(size=18, color='#212121'),
                             width=1200, height=1500,
                             plot_bgcolor='#eceff1', paper_bgcolor='#eceff1',
                             scene=dict(xaxis=dict(title=self.x_var.get(), gridcolor='gray', zerolinecolor='gray'),
                                      yaxis=dict(title=self.y_var.get(), gridcolor='gray', zerolinecolor='gray'),
                                      zaxis=dict(title=self.z_var.get(), gridcolor='gray', zerolinecolor='gray'),
                                      camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
                             font=dict(family="Helvetica", size=12, color='#212121'),
                             scene_aspectmode='cube')

            # Tab 2: PCA and 3D Surface for Ore Feed
            fig2 = make_subplots(rows=1, cols=2,
                               subplot_titles=('PCA of Mining Data', '3D Surface: Ore Feed'),
                               specs=[[{"type": "xy"}, {"type": "scene"}]])

            # Plot 4: PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(self.data.select_dtypes(include=[np.number]))
            fig2.add_trace(
                go.Scatter(x=pca_result[:, 0], y=pca_result[:, 1], mode='markers',
                          marker=dict(color=color_data_normalized, colorscale=cmap, opacity=alpha,
                                     size=6, line=dict(width=0.5, color='DarkSlateGrey'),
                                     colorbar=dict(title=self.color_var.get(), len=0.5, thickness=10)),
                          hovertemplate='PC1: %{x}<br>PC2: %{y}<br>Color: %{marker.color}<extra></extra>'),
                row=1, col=1
            )

            # Plot 5: 3D Surface for Ore Feed
            if self.check_data_variation(self.data['Ore Pulp Flow'], self.data['Starch Flow']):
                x_unique_flow = np.linspace(min(self.data['Ore Pulp Flow']), max(self.data['Ore Pulp Flow']), 10)
                y_unique_starch = np.linspace(min(self.data['Starch Flow']), max(self.data['Starch Flow']), 10)
                X_flow_grid, Y_starch_grid = np.meshgrid(x_unique_flow, y_unique_starch)
                try:
                    Z_flow_grid = griddata((self.data['Ore Pulp Flow'], self.data['Starch Flow']), z_data, (X_flow_grid, Y_starch_grid), method='cubic')
                    color_flow_grid = griddata((self.data['Ore Pulp Flow'], self.data['Starch Flow']), color_data, (X_flow_grid, Y_starch_grid), method='cubic')
                except:
                    Z_flow_grid = griddata((self.data['Ore Pulp Flow'], self.data['Starch Flow']), z_data, (X_flow_grid, Y_starch_grid), method='nearest')
                    color_flow_grid = griddata((self.data['Ore Pulp Flow'], self.data['Starch Flow']), color_data, (X_flow_grid, Y_starch_grid), method='nearest')
                color_flow_grid_normalized = (color_flow_grid - color_flow_grid.min()) / (color_flow_grid.max() - color_flow_grid.min())
                fig2.add_trace(
                    go.Surface(x=X_flow_grid, y=Y_starch_grid, z=Z_flow_grid, surfacecolor=color_flow_grid_normalized, colorscale=cmap, opacity=alpha,
                              showscale=True, colorbar=dict(title=self.color_var.get(), len=0.5, thickness=10),
                              contours=dict(z=dict(show=True, color='white', width=1)),  # Changed width to 1
                              lighting=dict(ambient=0.5, diffuse=0.8, specular=0.1)),
                    row=1, col=2
                )

            fig2.update_layout(title_text="Geological Data Visualization", title_font=dict(size=18, color='#212121'),
                             width=1400, height=600,
                             plot_bgcolor='#eceff1', paper_bgcolor='#eceff1',
                             scene=dict(xaxis=dict(title='Ore Pulp Flow', gridcolor='gray', zerolinecolor='gray'),
                                      yaxis=dict(title='Starch Flow', gridcolor='gray', zerolinecolor='gray'),
                                      zaxis=dict(title=self.z_var.get(), gridcolor='gray', zerolinecolor='gray'),
                                      camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
                             font=dict(family="Helvetica", size=12, color='#212121'),
                             scene_aspectmode='cube')

            # Save and open in browser with delay
            def open_browser():
                time.sleep(1)  # Delay to ensure plots are ready
                with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f1:
                    fig1.write_html(f1.name, config={'responsive': True})
                    webbrowser.open('file://' + os.path.realpath(f1.name))
                with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f2:
                    fig2.write_html(f2.name, config={'responsive': True})
                    webbrowser.open('file://' + os.path.realpath(f2.name))
                self.root.after(100, lambda: loading_label.destroy())  # Clear loading label

            open_browser()  # Removed threading to simplify, using delay instead

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plots: {str(e)}")
            loading_label.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GeologicalVisualizerApp(root)
    root.mainloop()