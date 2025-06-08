import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Language dictionaries
LANGUAGES = {
    'en': 'English',
    'hi': 'हिन्दी',
    'mr': 'मराठी'
}

UI_TEXT = {
    'en': {
        'title': '🌹 Rose Disease Detection System',
        'subtitle': 'For Farmers and Gardeners',
        'about': 'About',
        'instructions': 'Instructions',
        'tips': 'Tips',
        'upload_image': 'Upload Image',
        'choose_image': 'Choose an image...',
        'predict': '🔍 Predict Disease',
        'prediction_results': 'Prediction Results',
        'info': 'Information',
        'gallery': 'Disease Gallery',
        'common_conditions': 'Common rose conditions and their symptoms:',
        'supported': 'Supported Categories:',
        'tips_content': '• Take photos in good lighting\n• Focus on the affected area\n• Include both healthy and affected parts\n• Keep the image clear and steady',
        'no_image': 'Please upload an image to continue.',
        'confidence': 'Confidence',
        'desc': 'Description:',
        'recommend': 'Recommendations:'
    },
    'hi': {
        'title': '🌹 गुलाब रोग पहचान प्रणाली',
        'subtitle': 'किसानों और बागवानों के लिए',
        'about': 'परिचय',
        'instructions': 'निर्देश',
        'tips': 'सुझाव',
        'upload_image': 'छवि अपलोड करें',
        'choose_image': 'एक छवि चुनें...',
        'predict': '🔍 रोग पहचानें',
        'prediction_results': 'परिणाम',
        'info': 'जानकारी',
        'gallery': 'रोग गैलरी',
        'common_conditions': 'गुलाब की सामान्य स्थितियाँ और उनके लक्षण:',
        'supported': 'समर्थित श्रेणियाँ:',
        'tips_content': '• अच्छी रोशनी में फोटो लें\n• प्रभावित क्षेत्र पर ध्यान केंद्रित करें\n• स्वस्थ और प्रभावित दोनों भाग शामिल करें\n• छवि स्पष्ट रखें',
        'no_image': 'कृपया आगे बढ़ने के लिए एक छवि अपलोड करें।',
        'confidence': 'विश्वास',
        'desc': 'विवरण:',
        'recommend': 'सिफारिशें:'
    },
    'mr': {
        'title': '🌹 गुलाब रोग ओळख प्रणाली',
        'subtitle': 'शेतकरी आणि माळ्यांसाठी',
        'about': 'परिचय',
        'instructions': 'सूचना',
        'tips': 'टीप',
        'upload_image': 'प्रतिमा अपलोड करा',
        'choose_image': 'प्रतिमा निवडा...',
        'predict': '🔍 रोग ओळखा',
        'prediction_results': 'परिणाम',
        'info': 'माहिती',
        'gallery': 'रोग गॅलरी',
        'common_conditions': 'गुलाबाच्या सामान्य समस्या आणि त्यांची लक्षणे:',
        'supported': 'समर्थित श्रेण्या:',
        'tips_content': '• चांगल्या प्रकाशात फोटो घ्या\n• प्रभावित भागावर लक्ष केंद्रित करा\n• निरोगी आणि प्रभावित दोन्ही भाग समाविष्ट करा\n• प्रतिमा स्पष्ट ठेवा',
        'no_image': 'कृपया पुढे जाण्यासाठी प्रतिमा अपलोड करा.',
        'confidence': 'विश्वास',
        'desc': 'वर्णन:',
        'recommend': 'शिफारसी:'
    }
}

# Disease info in all languages
DISEASE_INFO = {
    'en': {
        'healthy': {
            'name': 'Healthy',
            'description': 'The rose leaf appears healthy with no signs of disease.',
            'remedy': '• Continue regular maintenance\n• Monitor for any changes\n• Maintain proper watering schedule\n• Keep good air circulation',
            'icon': '✅'
        },
        'downy_mildew': {
            'name': 'Downy Mildew',
            'description': 'Downy mildew is a fungal disease that appears as yellow patches on leaves with grayish-white mold underneath.',
            'remedy': '• Apply neem oil spray\n• Use baking soda solution (1 tbsp per gallon of water)\n• Improve air circulation\n• Remove infected leaves',
            'icon': '🍄'
        },
        'powdery_mildew': {
            'name': 'Powdery Mildew',
            'description': 'Powdery mildew appears as white powdery spots on leaves and stems.',
            'remedy': '• Spray with milk solution (1 part milk to 9 parts water)\n• Apply neem oil\n• Use baking soda spray\n• Prune affected areas',
            'icon': '❄️'
        },
        'black_spot': {
            'name': 'Black Spot',
            'description': 'Black spot causes black spots with yellow halos on leaves, leading to defoliation.',
            'remedy': '• Apply neem oil\n• Use baking soda solution\n• Remove infected leaves\n• Improve air circulation',
            'icon': '⚫'
        },
        'rose_slug': {
            'name': 'Rose Slug',
            'description': 'Rose slugs are sawfly larvae that skeletonize leaves.',
            'remedy': '• Handpick larvae\n• Apply neem oil\n• Use insecticidal soap\n• Encourage natural predators',
            'icon': '🐛'
        },
        'rose_mosaic': {
            'name': 'Rose Mosaic',
            'description': 'Rose mosaic virus causes yellow patterns on leaves.',
            'remedy': '• Remove infected plants\n• Use virus-free planting material\n• Maintain plant health\n• Control aphids',
            'icon': '🎨'
        },
        'rose_rust': {
            'name': 'Rose Rust',
            'description': 'Rose rust appears as orange powdery spots on leaves and stems.',
            'remedy': '• Apply neem oil\n• Use sulfur-based fungicide\n• Remove infected leaves\n• Improve air circulation',
            'icon': '🟠'
        }
    },
    'hi': {
        'healthy': {
            'name': 'स्वस्थ',
            'description': 'गुलाब की पत्ती स्वस्थ है और किसी रोग का कोई संकेत नहीं है।',
            'remedy': '• नियमित देखभाल जारी रखें\n• किसी भी बदलाव की निगरानी करें\n• उचित सिंचाई बनाए रखें\n• अच्छा वायु संचार रखें',
            'icon': '✅'
        },
        'downy_mildew': {
            'name': 'डाउनी मिल्ड्यू',
            'description': 'डाउनी मिल्ड्यू एक फफूंदजनित रोग है जो पत्तियों पर पीले धब्बों के रूप में दिखाई देता है, जिनके नीचे ग्रे-सफेद फफूंदी होती है।',
            'remedy': '• नीम का तेल स्प्रे करें\n• बेकिंग सोडा घोल (1 टेबलस्पून प्रति गैलन पानी) का उपयोग करें\n• वायु संचार सुधारें\n• संक्रमित पत्तियाँ हटाएँ',
            'icon': '🍄'
        },
        'powdery_mildew': {
            'name': 'पाउडरी मिल्ड्यू',
            'description': 'पाउडरी मिल्ड्यू पत्तियों और तनों पर सफेद पाउडर जैसे धब्बों के रूप में दिखाई देता है।',
            'remedy': '• दूध का घोल (1 भाग दूध, 9 भाग पानी) स्प्रे करें\n• नीम का तेल लगाएँ\n• बेकिंग सोडा स्प्रे करें\n• प्रभावित हिस्से काटें',
            'icon': '❄️'
        },
        'black_spot': {
            'name': 'ब्लैक स्पॉट',
            'description': 'ब्लैक स्पॉट पत्तियों पर काले धब्बे और पीले घेरे बनाता है, जिससे पत्तियाँ झड़ जाती हैं।',
            'remedy': '• नीम का तेल लगाएँ\n• बेकिंग सोडा घोल का उपयोग करें\n• संक्रमित पत्तियाँ हटाएँ\n• वायु संचार सुधारें',
            'icon': '⚫'
        },
        'rose_slug': {
            'name': 'रोज स्लग',
            'description': 'रोज स्लग पत्तियों को कंकाल जैसा बना देते हैं।',
            'remedy': '• लार्वा को हाथ से हटाएँ\n• नीम का तेल लगाएँ\n• कीटनाशक साबुन का उपयोग करें\n• प्राकृतिक शत्रुओं को प्रोत्साहित करें',
            'icon': '🐛'
        },
        'rose_mosaic': {
            'name': 'रोज मोज़ेक',
            'description': 'रोज मोज़ेक वायरस पत्तियों पर पीले पैटर्न बनाता है।',
            'remedy': '• संक्रमित पौधों को हटाएँ\n• वायरस-रहित पौध सामग्री का उपयोग करें\n• पौधों को स्वस्थ रखें\n• एफिड्स को नियंत्रित करें',
            'icon': '🎨'
        },
        'rose_rust': {
            'name': 'रोज रस्ट',
            'description': 'रोज रस्ट पत्तियों और तनों पर नारंगी पाउडर जैसे धब्बों के रूप में दिखाई देता है।',
            'remedy': '• नीम का तेल लगाएँ\n• सल्फर-आधारित फफूंदनाशक का उपयोग करें\n• संक्रमित पत्तियाँ हटाएँ\n• वायु संचार सुधारें',
            'icon': '🟠'
        }
    },
    'mr': {
        'healthy': {
            'name': 'निरोगी',
            'description': 'गुलाबाची पाने निरोगी आहेत आणि कोणत्याही रोगाचे लक्षण नाही.',
            'remedy': '• नियमित देखभाल सुरू ठेवा\n• कोणतेही बदल लक्षात घ्या\n• योग्य पाणी देणे सुरू ठेवा\n• चांगला वायुवीजन ठेवा',
            'icon': '✅'
        },
        'downy_mildew': {
            'name': 'डाउनी मिल्ड्यू',
            'description': 'डाउनी मिल्ड्यू हा एक बुरशीजन्य रोग आहे जो पानांवर पिवळ्या ठिपक्यांसारखा दिसतो आणि खाली राखाडी-श्वेत बुरशी असते.',
            'remedy': '• नीम तेलाचा फवारा करा\n• बेकिंग सोडा द्रावण (1 टेबलस्पून प्रति गॅलन पाणी) वापरा\n• वायुवीजन सुधारवा\n• संक्रमित पाने काढा',
            'icon': '🍄'
        },
        'powdery_mildew': {
            'name': 'पावडरी मिल्ड्यू',
            'description': 'पावडरी मिल्ड्यू पानांवर आणि खोडांवर पांढरे पावडर सारखे डाग निर्माण करतो.',
            'remedy': '• दूध द्रावण (1 भाग दूध, 9 भाग पाणी) फवारणी करा\n• नीम तेल लावा\n• बेकिंग सोडा फवारणी करा\n• प्रभावित भाग कापा',
            'icon': '❄️'
        },
        'black_spot': {
            'name': 'ब्लॅक स्पॉट',
            'description': 'ब्लॅक स्पॉट पानांवर काळे डाग आणि पिवळे वर्तुळे निर्माण करतो, ज्यामुळे पाने गळतात.',
            'remedy': '• नीम तेल लावा\n• बेकिंग सोडा द्रावण वापरा\n• संक्रमित पाने काढा\n• वायुवीजन सुधारवा',
            'icon': '⚫'
        },
        'rose_slug': {
            'name': 'रोज स्लग',
            'description': 'रोज स्लग पाने कंकालासारखी करतात.',
            'remedy': '• अळ्या हाताने काढा\n• नीम तेल लावा\n• कीटकनाशक साबण वापरा\n• नैसर्गिक शत्रूंना प्रोत्साहन द्या',
            'icon': '🐛'
        },
        'rose_mosaic': {
            'name': 'रोज मोझेक',
            'description': 'रोज मोझेक विषाणू पानांवर पिवळे नमुने तयार करतो.',
            'remedy': '• संक्रमित झाडे काढा\n• विषाणू-मुक्त लागवड साहित्य वापरा\n• झाडे निरोगी ठेवा\n• एफिड्स नियंत्रित करा',
            'icon': '🎨'
        },
        'rose_rust': {
            'name': 'रोज रस्ट',
            'description': 'रोज रस्ट पानांवर आणि खोडांवर नारिंगी पावडर सारखे डाग निर्माण करतो.',
            'remedy': '• नीम तेल लावा\n• सल्फर-आधारित बुरशीनाशक वापरा\n• संक्रमित पाने काढा\n• वायुवीजन सुधारवा',
            'icon': '🟠'
        }
    }
}

# Set page configuration
st.set_page_config(
    page_title="Rose Disease Detection",
    page_icon="🌹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #ffffff;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
    }
    .disease-info {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #ffffff;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #ffffff;
        margin: 1rem 0;
    }
    .sidebar-header {
        color: #4CAF50;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .disease-title {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: 600;
    }
    .confidence-medium {
        color: #FFA500;
        font-weight: 600;
    }
    .confidence-low {
        color: #FF4444;
        font-weight: 600;
    }
    .stExpander {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    try:
        model = tf.keras.models.load_model('custom_cnn_rose_disease_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure 'custom_cnn_rose_disease_model.h5' is in the correct directory.")
        return None

def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_disease_name(index):
    disease_mapping = {
        0: 'healthy',
        1: 'downy_mildew',
        2: 'powdery_mildew',
        3: 'black_spot',
        4: 'rose_slug',
        5: 'rose_mosaic',
        6: 'rose_rust'
    }
    return disease_mapping.get(index, 'unknown')

def get_confidence_class(confidence):
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    # Language selector
    lang = st.sidebar.selectbox('🌐 Language / भाषा / भाषा निवडा', list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])
    T = UI_TEXT[lang]
    DISEASES = DISEASE_INFO[lang]

    st.title(T['title'])
    st.subheader(T['subtitle'])
    
    # Sidebar
    st.sidebar.markdown(f'<div class="sidebar-header">{T["about"]}</div>', unsafe_allow_html=True)
    st.sidebar.write(f"""
    {T['supported']}
    - {DISEASES['healthy']['icon']} {DISEASES['healthy']['name']}
    - {DISEASES['downy_mildew']['icon']} {DISEASES['downy_mildew']['name']}
    - {DISEASES['powdery_mildew']['icon']} {DISEASES['powdery_mildew']['name']}
    - {DISEASES['black_spot']['icon']} {DISEASES['black_spot']['name']}
    - {DISEASES['rose_slug']['icon']} {DISEASES['rose_slug']['name']}
    - {DISEASES['rose_mosaic']['icon']} {DISEASES['rose_mosaic']['name']}
    - {DISEASES['rose_rust']['icon']} {DISEASES['rose_rust']['name']}
    """)
    st.sidebar.markdown(f'<div class="sidebar-header">{T["instructions"]}</div>', unsafe_allow_html=True)
    st.sidebar.write(f"""
    1. {T['upload_image']}
    2. {T['predict']}
    3. {T['prediction_results']} {T['info']}
    """)
    st.sidebar.markdown(f'<div class="sidebar-header">{T["tips"]}</div>', unsafe_allow_html=True)
    st.sidebar.write(T['tips_content'])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f'<div class="disease-title">{T["upload_image"]}</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(T['choose_image'], type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=T['choose_image'], use_column_width=True)
            if st.button(T['predict']):
                with st.spinner("Analyzing image..."):
                    img_array = np.array(image)
                    processed_img = preprocess_image(img_array)
                    model = load_model()
                    if model is not None:
                        prediction = model.predict(processed_img)
                        predicted_class = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class]
                        predicted_disease = get_disease_name(predicted_class)
                        confidence_class = get_confidence_class(confidence)
                        st.markdown(f"### {T['prediction_results']}")
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h4>{DISEASES[predicted_disease]['icon']} {DISEASES[predicted_disease]['name']}</h4>
                            <p class="{confidence_class}">{T['confidence']}: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"### {T['info']}")
                        st.markdown(f"""
                        <div class="disease-info">
                            <h4>{T['desc']}</h4>
                            <p>{DISEASES[predicted_disease]['description']}</p>
                            <h4>{T['recommend']}</h4>
                            <p>{DISEASES[predicted_disease]['remedy']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info(T['no_image'])
    with col2:
        st.markdown(f'<div class="disease-title">{T["gallery"]}</div>', unsafe_allow_html=True)
        st.write(T['common_conditions'])
        for disease, info in DISEASES.items():
            with st.expander(f"{info['icon']} {info['name']}"):
                st.write(f"**{T['desc']}** {info['description']}")
                st.write(f"**{T['recommend']}**")
                st.write(info['remedy'])

if __name__ == "__main__":
    main() 