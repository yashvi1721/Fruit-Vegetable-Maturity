import tkinter
from tkinter import filedialog as fd
from PIL import ImageTk, Image 
from tkinter import ttk 
import cv2

master = tkinter.Tk() 
master.configure(bg='#1b1c1c')
master.title("Fruit and VegeMaturity Detection")
var = tkinter.StringVar()
label = tkinter.Label( master, textvariable=var, fg = "blue",bg = "yellow",font = "Verdana 17 bold")
var.set("Fruit Maturity Detection")
label.pack()

def Train():
    argument=op.get()
    if argument=='Vegitable':
        import pathlib
        file = pathlib.Path("Model/Vegitable_360.h5")
        if file.exists ():
             tkinter.messagebox.showinfo(title="Done", message="Model Loaded")
        else:
            import keras
            from keras.preprocessing.image import ImageDataGenerator
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, Flatten
            from keras.layers import Conv2D, MaxPooling2D
            batch_size = 32
            num_classes = 16
            epochs = 50
            model_name = "Model/Vegitable_360.h5"
            path_to_train = "Datasets/Vegitable/Training"
            path_to_test = "Datasets/Vegitable/Test"        
            Generator = ImageDataGenerator()
            train_data = Generator.flow_from_directory(path_to_train, (100, 100), batch_size=batch_size)        
            test_data = Generator.flow_from_directory(path_to_test, (100, 100), batch_size=batch_size)
            model = Sequential()
            model.add(Conv2D(16, (5, 5), input_shape=(100, 100, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Conv2D(32, (5, 5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Conv2D(64, (5, 5),activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Conv2D(128, (5, 5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Flatten())
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.05))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.05))
            model.add(Dense(num_classes, activation="softmax"))
            model.summary()
            print('helllllllllllllllllllllllllllllllllllllllllllo')
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
            print('1111111111111111111111111111')
            model.fit(train_data,
                                steps_per_epoch=1000//batch_size,
                                epochs=epochs,
                                
                                verbose=1,
                                validation_data=test_data, validation_steps = 3)
            
            model.save(model_name)
            
    if argument=='Fruit':
        import pathlib
        file = pathlib.Path("Model/Fruit_360.h5")
        if file.exists ():
             tkinter.messagebox.showinfo(title="Done", message="Model Loaded")
        else:
            import keras
            from keras.preprocessing.image import ImageDataGenerator
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, Flatten
            from keras.layers import Conv2D, MaxPooling2D
            batch_size = 32
            num_classes = 21
            epochs = 50
            model_name = "Model/Fruit_360.h5"
            path_to_train = "Datasets/Fruit/Training"
            path_to_test = "Datasets/Fruit/Test"        
            Generator = ImageDataGenerator()
            train_data = Generator.flow_from_directory(path_to_train, (100, 100), batch_size=batch_size)        
            test_data = Generator.flow_from_directory(path_to_test, (100, 100), batch_size=batch_size)
            model = Sequential()
            model.add(Conv2D(16, (5, 5), input_shape=(100, 100, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Conv2D(32, (5, 5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Conv2D(64, (5, 5),activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Conv2D(128, (5, 5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.05))
            model.add(Flatten())
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.05))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.05))
            model.add(Dense(num_classes, activation="softmax"))
            model.summary()
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
            model.fit_generator(train_data,
                                steps_per_epoch=1000//batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=test_data, validation_steps = 3)
            model.save(model_name)
            master.mainloop()  
            
def CaptureImage():
    import cv2
    camera = cv2.VideoCapture(0)
    while 1:
        return_value, image = camera.read()
        cv2.imshow("Frame",image)
        image=cv2.resize(image, (100, 100))
        cv2.imwrite("Model/query.jpg", image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    camera.release()
    del(camera)
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows()
    img = Image.open("Model/query.jpg") 
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)    
    panel = tkinter.Label(master, image = img)       
    panel.image = img 
    panel.place(x = 180, y = 5 + 1*30, width=250, height=250)
    
def BowseImage():
    name= fd.askopenfilename()
    img = Image.open(name) 
    img.save("Model/query.jpg")
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)    
    panel = tkinter.Label(master, image = img)       
    panel.image = img 
    panel.place(x = 180, y = 5 + 1*30, width=250, height=250)
    master.mainloop()    

def medfilt():
    img = cv2.imread("Model/query.jpg")
    img=cv2.medianBlur(img, 3)
    cv2.imwrite("Model/Pre.jpg",img)
    img = Image.open("Model/Pre.jpg")
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)   
    panel = tkinter.Label(master, image = img)       
    panel.image = img 
    panel.place(x = 460, y = 5 + 1*30, width=250, height=250)    
    master.mainloop()   

def ApplyCNN():
    import cv2
    import numpy as np
    from keras.models import load_model

    # Load the image
    img = cv2.imread("Model/query.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Convert to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Calculate average hue for ripe fruit
    average_hue = np.mean(hsv_img[:, :, 0])

    # Define threshold range for ripe fruit (adjust as needed)
    lower_range = np.array([20, 50, 50])
    upper_range = np.array([30, 255, 255])

    # Create mask for pixels within the threshold range
    mask = cv2.inRange(hsv_img, lower_range, upper_range)

    # Count pixels within the mask
    masked_pixels = cv2.countNonZero(mask)

    # Calculate total number of pixels
    total_pixels = np.prod(mask.shape[:2])

    # Calculate percentage of ripe fruit pixels
    ripe_percentage = (masked_pixels / total_pixels) * 100

    # Define maturity levels based on average hue
    if average_hue >= 0 and average_hue < 30:
        maturity_level = "Unripe"
    elif average_hue >= 30 and average_hue < 60:
        maturity_level = "Semi-Ripe"
    else:
        maturity_level = "Ripe"

    # Load the CNN model
    argument = op.get()
    if argument == 'Vegitable':
        model = load_model('Model/Vegitable_360.h5')
    elif argument == 'Fruit':
        model = load_model('Model/Fruit_360.h5')

    # Reshape the image for model prediction
    img = cv2.resize(img, (100, 100))
    img = img.reshape(1, 100, 100, 3)
    argument=op.get()

    # Perform model prediction
    result = model.predict(img)

    # Define class labels
    if argument == 'Vegitable':
        model = load_model('Model/Vegitable_360.h5')
        model.summary()
        result=model.predict(img)
        labels = {
            0: 'Beetroot',
            1: 'Capsicum',
            2: 'Cauliflower',
            3: 'Corn',
            4: 'Corn Husk',
            5: 'Eggplant',
            6: 'Fig',
            7: 'Ginger Root',
            8: 'Kohlrabi',
            9: 'Lemon',
            10: 'Limes',
            11: 'Onion Red',
            12: 'Onion White',
            13: 'Potato White',
            14: 'Tomato Heart',
            15: 'Tomato Maroon'
        }
        labels_d=list(labels)
        result_classes = result.argmax(axis=-1)
        Class=label_d[result_classes[0]]
    elif argument == 'Fruit':
        model = load_model('Model/Fruit_360.h5')
        model.summary()
        labels = {
            0: 'Apple Braeburn',
            1: 'Banana',
            2: 'Cherry 2',
            3: 'Cocos',
            4: 'Dates',
            5: 'Grape White',
            6: 'Guava',
            7: 'Hazelnut',
            8: 'Huckleberry',
            9: 'Kiwi',
            10: 'Lemon Meyer',
            11: 'Lychee',
            12: 'Mango Green',
            13: 'Mango Red',
            14: 'Orange',
            15: 'Papaya',
            16: 'Raspberry',
            17: 'Redcurrant',
            18: 'Strawberry',
            19: 'Walnut',
            20: 'Watermelon'
        }
        labels_d=list(labels)
        result=model.predict(img)
        result_classes = result.argmax(axis=-1)
        Class=labels_d[result_classes[0]]

    # Get the predicted class label
    predicted_class = labels[np.argmax(result)]

    # Display the ripeness, maturity level, and predicted class labels
    result_var = tkinter.StringVar()
    result_label = tkinter.Label(master, textvariable=result_var, fg="yellow", bg="red", font="Verdana 10 bold")
    result_var.set(f"Ripe Percentage: {ripe_percentage:.2f}%, Maturity Level: {maturity_level}, Predicted Class: {predicted_class}")
    result_label.place(x=10, y=180 + 8 * 30, width=400, height=50)

    master.mainloop()

    import cv2
    import numpy as np
    from keras.models import load_model

    # Load the image
    img = cv2.imread("Model/query.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Convert to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Calculate average hue for ripe fruit
    average_hue = np.mean(hsv_img[:, :, 0])

    # Define threshold range for ripe fruit (adjust as needed)
    lower_range = np.array([20, 50, 50])
    upper_range = np.array([30, 255, 255])

    # Create mask for pixels within the threshold range
    mask = cv2.inRange(hsv_img, lower_range, upper_range)

    # Count pixels within the mask
    masked_pixels = cv2.countNonZero(mask)

    # Calculate total number of pixels
    total_pixels = np.prod(mask.shape[:2])

    # Calculate percentage of ripe fruit pixels
    ripe_percentage = (masked_pixels / total_pixels) * 100

    # Define maturity levels based on average hue
    if average_hue >= 0 and average_hue < 30:
        maturity_level = "Unripe"
    elif average_hue >= 30 and average_hue < 60:
        maturity_level = "Semi-Ripe"
    else:
        maturity_level = "Ripe"

    # Load the CNN model
    argument = op.get()
    if argument == 'Vegitable':
        model = load_model('Model/Vegitable_360.h5')
    elif argument == 'Fruit':
        model = load_model('Model/Fruit_360.h5')

    # Reshape the image for model prediction
    img = cv2.resize(img, (100, 100))
    img = img.reshape(1, 100, 100, 3)

    # Perform model prediction
    result = model.predict(img)

    # Define class labels
    if argument == 'Vegitable':
        labels = {
            0: 'Beetroot',
            1: 'Capsicum',
            2: 'Cauliflower',
            3: 'Corn',
            4: 'Corn Husk',
            5: 'Eggplant',
            6: 'Fig',
            7: 'Ginger Root',
            8: 'Kohlrabi',
            9: 'Lemon',
            10: 'Limes',
            11: 'Onion Red',
            12: 'Onion White',
            13: 'Potato White',
            14: 'Tomato Heart',
            15: 'Tomato Maroon'
        }
    elif argument == 'Fruit':
        labels = {
            0: 'Apple Braeburn',
            1: 'Banana',
            2: 'Cherry 2',
            3: 'Cocos',
            4: 'Dates',
            5: 'Grape White',
            6: 'Guava',
            7: 'Hazelnut',
            8: 'Huckleberry',
            9: 'Kiwi',
            10: 'Lemon Meyer',
            11: 'Lychee',
            12: 'Mango Green',
            13: 'Mango Red',
            14: 'Orange',
            15: 'Papaya',
            16: 'Raspberry',
            17: 'Redcurrant',
            18: 'Strawberry',
            19: 'Walnut',
            20: 'Watermelon'
        }

    # Get the predicted class label
    predicted_class = labels[np.argmax(result)]

    # Display the ripeness, maturity level, and predicted class labels
    ripeness_var = tkinter.StringVar()
    maturity_var = tkinter.StringVar()
    ripeness_label = tkinter.Label(master, textvariable=ripeness_var, fg="yellow", bg="red", font="Verdana 10 bold")
    ripeness_var.set(f"Ripeness: {ripe_percentage:.2f}%, Maturity Level: {maturity_level}, Predicted Class: {predicted_class}")
    ripeness_label.place(x=10, y=180 + 8 * 30, width=400, height=50)

    master.mainloop()

    import cv2
    from keras.models import load_model
    img = cv2.imread("Model/query.jpg")
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img=img.reshape(1,100,100,3)
    argument=op.get()
    if argument=='Vegitable':
        model = load_model('Model/Vegitable_360.h5')
        model.summary()
        result=model.predict(img)
        label = {'Beetroot': 0,
                 'Capsicum':1,
                 'Cauliflower': 2,
                 'Corn':3,
                 'Corn Husk':4,
                 'Eggplant':5,
                 'Fig':6,
                 'Ginger Root':7,
                 'Kohlrabi':8,
                 'Lemon':9,
                 'Limes':10,
                 'Onion Red': 11,
                 'Onion White':12,
                 'Potato White':13,
                 'Tomato Heart':14,
                 'Tomato Maroon':15}  
        label_d=list(label)
        result_classes = result.argmax(axis=-1)
        Class=label_d[result_classes[0]]
    if argument=='Fruit':
        model = load_model('Model/Fruit_360.h5')
        model.summary()
        label =  {'Apple Braeburn': 0,
                  'Banana': 1,
                  'Cherry 2':2,
                  'Cocos':3,
                  'Dates':4,
                  'Grape White':5,
                  'Guava':6,
                  'Hazelnut':7,
                  'Huckleberry':8,
                  'Kiwi':9,
                  'Lemon Meyer':10,
                  'Lychee':11,
                  'Mango Green': 12,
                  'Mango Red':13,
                  'Orange':14,
                  'Papaya':15,
                  'Raspberry':16,
                  'Redcurrant':17,
                  'Strawberry':18,
                  'Walnut':19,
                  'Watermelon':20}  
        label_d=list(label)
        result=model.predict(img)
        result_classes = result.argmax(axis=-1)
        Class=label_d[result_classes[0]]
    var = tkinter.StringVar()
    label = tkinter.Label( master, textvariable=var, fg = "yellow",bg = "red",font = "Verdana 10 bold")
    var.set(Class)
    label.place(x = 10, y = 180 + 8*30, width=150, height=50)    
    master.mainloop()
 


   
def Exit():
    master.destroy()

master.geometry("750x500+100+100") 
master.resizable(width = True, height = True) 

op = ttk.Combobox(master,values=["Vegitable","Fruit"],font = "Verdana 10 bold")
op.place(x = 10, y = 5 + 1*30, width=150, height=50)
op.current(0)

b0 = tkinter.Button(master, text = "Train/Load Model", command = Train,bg='#F1EAE3',fg='black',font = "Verdana 10 bold") 
b0.place(x = 10, y = 30 + 2*30, width=150, height=50)

b1 = tkinter.Button(master, text = "Capture Image", command = CaptureImage,bg='#F1EAE3',fg='black',font = "Verdana 10 bold") 
b1.place(x = 10, y = 55 + 3*30, width=150, height=50)

b1 = tkinter.Button(master, text = "Upload Image", command = BowseImage,bg='#F1EAE3',fg='black',font = "Verdana 10 bold") 
b1.place(x = 10, y = 80 + 4*30, width=150, height=50)

b2 = tkinter.Button(master, text = "Pre-Process", command = medfilt,bg='#F1EAE3',fg='black',font = "Verdana 9 bold") 
b2.place(x = 10, y = 105 + 5*30, width=150, height=50)


b3 = tkinter.Button(master, text = "CNN Recognization", command = ApplyCNN,bg='#F1EAE3',fg='black',font = "Verdana 10 bold") 
b3.place(x = 10, y = 130 + 6*30, width=150, height=50)

b4 = tkinter.Button(master, text = "Quit", command = Exit,bg='#F1EAE3',fg='black',font = "Verdana 10 bold") 
b4.place(x = 10, y = 155 + 7*30, width=150, height=50)

var = tkinter.StringVar()
label = tkinter.Label( master, textvariable=var, fg = "yellow",bg = "red",font = "Verdana 10 bold")
var.set("Class Name")
label.place(x = 10, y = 180 + 8*30, width=150, height=50)

master.mainloop() 
