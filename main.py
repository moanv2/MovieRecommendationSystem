import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from CBF import Content_Based_Filtering

"""Get content based filtering file"""
cbf = Content_Based_Filtering("imdb_movie_dataset.csv")
cbf.pre_process()
cbf.generate_count_matrix()


def recommend_movies():
    # Get the user input
    title = movie_entry.get()

    # Clear the previous results
    recommendations_list.delete(0, tk.END)

    # Errorhandling so code does not crash
    try:
        # Get recommendations
        recommendations = cbf.recommend(title, top_n=10)

        if not recommendations:
            messagebox.showerror("Error", f"No recommendations found for '{title}'!")
            return

        # Display recommendations in the listbox
        for movie in recommendations:
            recommendations_list.insert(tk.END, movie)
    except Exception as e:
        messagebox.showerror("Error", str(e))


# Main tkinter window
root = tk.Tk()
root.title("Movie Recommendation System")

# Create a label and entry for movie input
movie_label = tk.Label(root, text="Enter Movie Title:")
movie_label.pack(pady=10)

movie_entry = tk.Entry(root, width=40)
movie_entry.pack(pady=5)

# Create a button to get recommendations
recommend_button = tk.Button(root, text="Get Recommendations", command=recommend_movies)
recommend_button.pack(pady=10)

# Create a listbox to display recommendations
recommendations_list = tk.Listbox(root, width=50, height=15)
recommendations_list.pack(pady=10)

# Run the tkinter main loop
root.mainloop()