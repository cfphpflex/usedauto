from predict import predict_price

def test_prediction():
    # Sample vehicle data
    vehicle = {
        'year': 2020,
        'manufacturer': 'tesla',
        'model': 'model 3',
        'condition': 'excellent',
        'odometer': 25000,
        'fuel': 'electric',
        'title_status': 'clean',
        'transmission': 'automatic',
        'drive': 'rwd',
        'type': 'sedan',
        'paint_color': 'white'
    }
    
    print("\nStarting prediction test...")
    print(f"Test vehicle: {vehicle}")
    
    try:
        predicted_price = predict_price(vehicle)
        if predicted_price is not None:
            print(f"\nSuccess! Predicted price: ${predicted_price:,.2f}")
        else:
            print("\nFailed to predict price")
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")

if __name__ == '__main__':
    test_prediction() 