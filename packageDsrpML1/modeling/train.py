from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

class MLProc:

    def __init__(self, model_name:str, data):
        self.model_name = model_name
        self.data = data
      
    def pre_process_data(self):
        self.data["is_genuine"] = self.data["is_genuine"].astype(int)
        self.data.dropna(inplace=True)
        print("Vistazo data depurada:")
        print("----------------------")
        print(self.data.info())
        print("\n\n")
        
    def split_data(self):
        self.pre_process_data()
        x = self.data.drop("is_genuine", axis=1)
        y = self.data["is_genuine"]
        print("1.Separando datos de train y test...")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, random_state=100, test_size=0.2)
        
    def train(self, model):        
        self.split_data()
        print(f"2.Entrenando model {self.model_name}")
        self.fitted_model = model.fit(self.x_train, self.y_train)
        print("3.Entrenamiento Finalizado....")

    def predict(self):
        self.predictions = self.fitted_model.predict(self.x_test)

    def metricas(self):
        print("*********************************************************************")
        print(f"Mostrando Metricas pre-establecidas del modelo {self.model_name}")
        print("*********************************************************************")
        nombre = type(self.fitted_model).__name__

        if nombre == "LogisticRegression":
            #print(self.predictions)
            print(classification_report(self.predictions, self.y_test))
            cm = confusion_matrix(self.y_test, self.predictions)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            disp.ax_.set_title("Matriz de Confusión: Logistic Regression")
        elif nombre == "LinearRegression":
            print(nombre)
        elif nombre == "PCA":
            print(nombre)
        elif model_name == "KMeans":
             print(nombre)
        else:
            print("Modelo no reconocido.")
        

    