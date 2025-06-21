from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

def metricas(predicciones, y_test, model):
    print("*********************************************************************")
    print(f"Mostrando Metricas pre-establecidas del modelo {type(model).__name__}")
    print("*********************************************************************")
    
    #print(self.predictions)
    print(classification_report(predicciones, y_test))
    cm = confusion_matrix(y_test, predicciones)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    disp.ax_.set_title(f"Matriz de Confusión: {type(model).__name__}")

class MLProc:

    def __init__(self, c_model_name:str, model, data):
        self.c_model_name = c_model_name
        self.model = model
        self.data = data
        self.model_name = type(model).__name__
      
    def pre_process_data(self):
        self.data["is_genuine"] = self.data["is_genuine"].astype(int)        
        self.data.dropna(inplace=True)
        
        print("************************")
        print("Vistazo data depurada:")
        print("************************")
        
        print(self.data.info())
        print("\n\n")
        
    def split_data(self):
        self.pre_process_data()
        
        x = self.data.drop("is_genuine", axis=1)
        y = self.data["is_genuine"]
        
        print("1.Separando datos de train y test...")
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, random_state=100, test_size=0.2)
        
    def train(self):
        self.split_data()        
      
        print(f"2.Entrenando model {self.c_model_name}")
        self.fitted_model = self.model.fit(self.x_train, self.y_train)

        print("3.Entrenamiento Finalizado....")

    def predict(self):
        self.predictions = self.fitted_model.predict(self.x_test)



    