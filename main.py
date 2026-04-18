from ui import build_main_window # imports from ui.py

ui = build_main_window() # assigns the function from ui to this 
window = ui["window"] # pulls the main Tkinter window out of the dictionary

status_text = ui["status_text"]


window.mainloop() # keeps the window running
