from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from CleanData import *
from DecisionTree import *
from LogisticRegressionModel import *
from KNN import *

stats = CleanData()
DTData = DecisionTreeModel(1)
KNNData = KNN(1)
LogRData = LogisticRegression(1)

def switch_frame(page_name):
    for widget in main_content.winfo_children():
        widget.destroy()
    
    if page_name == "Home":

        logo_section = Frame(main_content, bg="white")
        logo_section.pack(fill=X,padx=20,pady=30)

        Label(logo_section, image=ASU_IMAGE,bg="white").pack(side=LEFT)
        Label(logo_section, image=FCIS_IMAGE,bg="white").pack(side=RIGHT)

        Label(main_content, text="ROB #1 AI Project", font=('Arial', 40, 'bold'), fg="#333").pack(pady=(0,50))
        Label(main_content, text="Welcome to the AI Dashboard.\nSelect a tool from the left to begin.", font=('Arial', 14)).pack()
        Label(main_content, text="Team members:", font=('Arial', 20)).pack(pady=(20,0))
        Label(main_content, text="Aly Amr Ibrahim – 20241701504\nLogy Ahmed Talaat – 20241701506\nMaritsia Ligor Khalaf – 20241701507\nToka Abdelwahab Fathy – 20231700521", font=('Arial', 14)).pack(pady=(10,0))
        
        Label(main_content, text="Supervised by:", font=('Arial', 20)).pack(pady=(20,0))
        Label(main_content, text="Dr. Dina Elsayed\nTA: Mohammed Magdy", font=('Arial', 14)).pack(pady=(10,0))

    elif page_name == "Pre Processing":
        Label(main_content, text="Pre Processing", font=('Arial', 30, 'bold')).pack(pady=20)
        controls_frame = Frame(main_content, bg="white")
        controls_frame.pack(fill=X, padx=20, pady=5)

        Label(controls_frame, text="Before Cleaning", font=('Arial', 20, 'bold')).pack(pady=20)
        Label(controls_frame, text=f"Number of rows in test dataset: {stats["numOfUncleanRowsInTest"]}", font=('Arial', 12), bg="white").pack()
        
        
        Label(controls_frame, text=f"Number of columns in test dataset: {stats["numOfUncleanCols"]}", font=('Arial', 12), bg="white").pack()
        Label(controls_frame, bg="white").pack(pady=10)

        Label(controls_frame, text=f"Number of rows in train dataset: {stats["numOfUncleanRowsInTrain"]}", font=('Arial', 12), bg="white").pack()
        
        
        Label(controls_frame, text=f"Number of columns in train dataset: {stats["numOfUncleanCols"]}", font=('Arial', 12), bg="white").pack()
        

        Label(controls_frame, text="After Cleaning", font=('Arial', 20, 'bold')).pack(pady=20)
        Label(controls_frame, text=f"Number of rows in test dataset: {stats["numOfCleanRowsInTest"]}", font=('Arial', 12), bg="white").pack()
        
        
        Label(controls_frame, text=f"Number of columns in test dataset: {stats["numOfUncleanCols"]}", font=('Arial', 12), bg="white").pack()
        Label(controls_frame, bg="white").pack(pady=10)

        Label(controls_frame, text=f"Number of rows in train dataset: {stats["numOfCleanRowsInTrain"]}", font=('Arial', 12), bg="white").pack()
        
        
        Label(controls_frame, text=f"Number of columns in train dataset: {stats["numOfUncleanCols"]}", font=('Arial', 12), bg="white").pack()

        Label(controls_frame, bg="white").pack(pady=10)

        Label(controls_frame, text=f"Columns Dropped: {stats["droppedCols"]}", font=('Arial', 12), bg="white").pack()  

    elif page_name == "Models":
        Label(main_content, text="Models", font=('Arial', 30, 'bold'), fg="#333").pack(pady=20)

        cards_container = Frame(main_content, bg="white")
        cards_container.pack(expand=True, fill=BOTH, padx=50, pady=20)

        card_frame = Frame(cards_container, bg="white", padx=20)
        card_frame.pack(side=LEFT, expand=True, fill=BOTH)

        display_box = Label(card_frame, 
                            text=f"Decision Tree\n\nAccuracy: {DTData["accuracy"]}\nPrecision: {DTData["precision"]}\nRecall: {DTData["recall"]}\nF1 Score: {DTData["f1"]}\n\nBest Hyperparameters:\nMax Depth: {DTData["bestMaxDepth"]}\nMin Samples Split: {DTData["bestSamplesSplit"]}", 
                            font=('Arial', 14), 
                            bg="#f0f0f0", 
                            width=20, 
                            height=12, 
                            relief="solid", 
                            bd=1)
        display_box.pack(pady=(0, 20))
        action_btn = Button(card_frame, 
                            text=f"Show Visualisations", 
                            font=('Arial', 12, 'bold'), 
                            bg="#2196F3", 
                            fg="white", 
                            width=18, 
                            pady=5,
                            command=DTData["graphs"])
        action_btn.pack()

        card_frame = Frame(cards_container, bg="white", padx=20)
        card_frame.pack(side=LEFT, expand=True, fill=BOTH)

        display_box = Label(card_frame, 
                            text=f"k-Nearest Neighbor\n\nAccuracy: {KNNData["accuracy"]}\nPrecision: {KNNData["precision"]}\nRecall: {KNNData["recall"]}\nF1 Score: {KNNData["f1"]}\n\nBest Hyperparameters:\nK: {KNNData["bestK"]}\nWeights: {KNNData["bestWeight"]}\nMetric: {KNNData["bestMetric"]}\nBest CV Score: {KNNData["bestCV"]}", 
                            font=('Arial', 14), 
                            bg="#f0f0f0", 
                            width=20, 
                            height=13, 
                            relief="solid", 
                            bd=1)
        display_box.pack(pady=(0, 20))
        action_btn = Button(card_frame, 
                            text=f"Show Visualisations", 
                            font=('Arial', 12, 'bold'), 
                            bg="#2196F3", 
                            fg="white", 
                            width=18, 
                            pady=5, 
                            command=KNNData["graphs"])
        action_btn.pack()

        card_frame = Frame(cards_container, bg="white", padx=20)
        card_frame.pack(side=LEFT, expand=True, fill=BOTH)

        display_box = Label(card_frame, 
                            text=f"Logistic Regression\n\nAccuracy: {LogRData["accuracy"]}\nPrecision: {LogRData["precision"]}\nRecall: {LogRData["recall"]}\nF1 Score: {LogRData["f1"]}\n\nBest Hyperparameters:\nC: {LogRData["bestC"]}\nMax Iterations: {LogRData["bestMaxIter"]}",
                            font=('Arial', 14), 
                            bg="#f0f0f0", 
                            width=20, 
                            height=12, 
                            relief="solid", 
                            bd=1)
        display_box.pack(pady=(0, 20))
        action_btn = Button(card_frame, 
                            text=f"Show Visualisations", 
                            font=('Arial', 12, 'bold'), 
                            bg="#2196F3", 
                            fg="white", 
                            width=18, 
                            pady=5,
                            command=LogRData["graphs"])
        action_btn.pack()

    elif page_name == "Prediction":

        dtvar = 0
        knnvar = 0
        logvar = 0

        Label(main_content, text="Prediction", font=('Arial', 30, 'bold'), fg="#333").pack(pady=20)
        Label(main_content, text="Enter a Test Row Index to predict.", font=('Arial', 12)).pack(pady=(0, 20))

        cards_container = Frame(main_content, bg="white")
        cards_container.pack(expand=True, fill=BOTH, padx=50, pady=20)

        card_frame = Frame(cards_container, bg="white", padx=20)
        card_frame.pack(side=LEFT, expand=True, fill=BOTH)
        
        display_box = Label(card_frame, 
                            text=f"Decision Tree\n\nAccuracy: {DTData["accuracy"]}", 
                            font=('Arial', 14), 
                            bg="#f0f0f0", 
                            width=20, 
                            height=10, 
                            relief="solid", 
                            bd=1)
        display_box.pack()

        def decTree():
            dec_user_input = dec_entry_box.get()
            if not dec_user_input.isdigit():
                messagebox.showerror("Wrong input", "Please enter a number from 0-152")
                return
            dec_user_input = int(dec_user_input)
            if dec_user_input>152 or dec_user_input<0:
                messagebox.showerror("Wrong input", "Please enter a number from 0-152")
                return
            DTData = DecisionTreeModel(dec_user_input)
            messagebox.showinfo(f"Decison Tree Prediction! ({dec_user_input})", f"Predicted value: {"Expensive" if DTData["prediction"] else "Not-Expensive"}\nReal value: {"Expensive" if DTData["actual"] else "Not-Expensive"}" )

        dec_entry_box = Entry(card_frame, width=10, font=('Arial', 18), state="normal")
        dec_entry_box.pack(pady=(20))
        dec_entry_box.focus_set()
        Button(card_frame, text="Predict DT", font=('Arial', 12, 'bold'), bg="#2196F3", fg="white", 
               width=18, pady=5, command=decTree).pack()
        
        card_frame = Frame(cards_container, bg="white", padx=20)
        card_frame.pack(side=LEFT, expand=True, fill=BOTH)
        
        display_box = Label(card_frame, 
                            text=f"k-Nearest Neighbor\n\nAccuracy: {KNNData["accuracy"]}", 
                            font=('Arial', 14), 
                            bg="#f0f0f0", 
                            width=20, 
                            height=10, 
                            relief="solid", 
                            bd=1)
        display_box.pack()

        def knn():
            knn_user_input = knn_entry_box.get()
            if not knn_user_input.isdigit():
                messagebox.showerror("Wrong input", "Please enter a number from 0-152")
                return
            knn_user_input = int(knn_user_input)
            if knn_user_input>152 or knn_user_input<0:
                messagebox.showerror("Wrong input", "Please enter a number from 0-152")
                return
            KNNData = KNN(knn_user_input)
            messagebox.showinfo(f"KNN Prediction! ({knn_user_input})", f"Predicted value: {"Expensive" if KNNData["prediction"] else "Not-Expensive"}\nReal value: {"Expensive" if KNNData["actual"] else "Not-Expensive"}" )

        knn_entry_box = Entry(card_frame, width=10, font=('Arial', 18), state="normal")
        knn_entry_box.pack(pady=(20))
        knn_entry_box.focus_set()
        Button(card_frame, text="Predict DT", font=('Arial', 12, 'bold'), bg="#2196F3", fg="white", 
               width=18, pady=5, command=knn).pack()
        
        card_frame = Frame(cards_container, bg="white", padx=20)
        card_frame.pack(side=LEFT, expand=True, fill=BOTH)
        
        display_box = Label(card_frame, 
                            text=f"Logistic Regression\n\nAccuracy: {LogRData["accuracy"]}", 
                            font=('Arial', 14), 
                            bg="#f0f0f0", 
                            width=20, 
                            height=10, 
                            relief="solid", 
                            bd=1)
        display_box.pack()

        def logReg():
            log_user_input = log_entry_box.get()
            if not log_user_input.isdigit():
                messagebox.showerror("Wrong input", "Please enter a number from 0-152")
                return
            log_user_input = int(log_user_input)
            if log_user_input>152 or log_user_input<0:
                messagebox.showerror("Wrong input", "Please enter a number from 0-152")
                return
            LogRData = LogisticRegression(log_user_input)
            messagebox.showinfo(f"Logistic Regression Prediction! ({log_user_input})", f"Predicted value: {"Expensive" if LogRData["prediction"] else "Not-Expensive"}\nReal value: {"Expensive" if LogRData["actual"] else "Not-Expensive"}" )

        log_entry_box = Entry(card_frame, width=10, font=('Arial', 18), state="normal")
        log_entry_box.pack(pady=(20))
        log_entry_box.focus_set()
        window.update()
        Button(card_frame, text="Predict DT", font=('Arial', 12, 'bold'), bg="#2196F3", fg="white", 
               width=18, pady=5, command=logReg).pack()

window = Tk()
window.geometry("1280x720")
window.title("ROB #1 AI Project")

sidebar = Frame(window, bg="#e0e0e0", width=250)
sidebar.pack(side=LEFT, fill=Y)
sidebar.pack_propagate(False) 

main_content = Frame(window, bg="white")
main_content.pack(side=RIGHT, fill=BOTH, expand=True)

FCIS_IMAGE = PhotoImage(file="images/FCIS.png")
ASU_IMAGE = PhotoImage(file="images/ASU.png")

def create_nav_btn(text):
    return Button(sidebar, 
                  text=text, 
                  font=('Arial', 14), 
                  bg="#dddddd", 
                  bd=0, 
                  pady=10, 
                  command=lambda: switch_frame(text))

btn_home = create_nav_btn("Home")
btn_home.pack(fill=X, pady=5, padx=10)

btn_clean = create_nav_btn("Pre Processing")
btn_clean.pack(fill=X, pady=5, padx=10)

btn_ai = create_nav_btn("Models")
btn_ai.pack(fill=X, pady=5, padx=10)

btn_graphs = create_nav_btn("Prediction")
btn_graphs.pack(fill=X, pady=5, padx=10)

switch_frame("Prediction")

messagebox.showinfo("Alert", "Data Cleaned and saved in \"CleanData\" Folder")

window.mainloop()