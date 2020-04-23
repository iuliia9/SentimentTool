import tkinter as tk
import SentimentAnalysis

def get_input():
    keyword = entry.get()
    SentimentAnalysis.main(keyword)

m = tk.Tk()
m.title('Sentiment Tool')
tk.Label(m, text='Key Word').grid(row=0)
entry = tk.Entry(m)
entry.grid(row=0, column=1)
tk.Button(m, text='Exit',command=m.quit).grid(row=3,column=0,sticky=tk.W,pady=4)
tk.Button(m, text='Search', command=get_input).grid(row=3, column=1, sticky=tk.W,pady=4)
m.mainloop()