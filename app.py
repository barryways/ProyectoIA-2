from modelo.obtener_pred import predecir_letra

if __name__ == "__main__":
    resultado = predecir_letra("asl/asl_alphabet_test/asl_alphabet_test/D_test.jpg")
    print(f"La letra es: {resultado}")