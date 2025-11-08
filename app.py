# ...existing code...
import streamlit as st
import numpy as np
import pickle
import os
import traceback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# --- Configuration ---
# !! REPLACE THESE PLACEHOLDERS !!
# Use absolute paths based on the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, 'smart_closet_model.pkl')  # e.g., 'linear_reg.pkl'
NPY_FILE = os.path.join(SCRIPT_DIR, 'image_paths.npy')    # e.g., 'scaler_minmax.npy'
APP_TITLE = "My ML App using NPY Data"

st.set_page_config(page_title=APP_TITLE)
st.title(f"🚀 {APP_TITLE}")
st.write("---")

# --- Simple fallback model so the app still works if real model fails to load ---
class DummyModel:
    def predict(self, X):
        # return zeros with correct shape
        X = np.asarray(X)
        out = np.zeros((X.shape[0],))
        return out

# --- Function to load assets (Model and NPY data) ---
# @st.cache_resource ensures the large files are loaded only ONCE,
# making your app fast.
@st.cache_resource
def load_assets():
    model = None
    npy_data = None
    # Try to load model
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Check if it's a dictionary (FAISS index data)
            if isinstance(loaded_data, dict):
                st.info(f"Loaded dictionary with keys: {list(loaded_data.keys())}")
                # If it's a dict, it might contain 'faiss_index' or 'model'
                if 'model' in loaded_data:
                    model = loaded_data['model']
                elif 'faiss_index' in loaded_data:
                    # This is FAISS data, wrap it for compatibility
                    model = loaded_data  # Keep the whole dict as model
                else:
                    model = loaded_data
            else:
                model = loaded_data
                
        except Exception as e:
            # Provide concise actionable guidance for common pyarrow/C++ build errors
            st.error("Failed to load the model file. The pickle load raised an exception.")
            st.text(f"Error: {e}")
            st.text("If the error mentions 'pyarrow' or 'cmake', you may need a prebuilt wheel or conda:")
            st.code("Use conda: conda install -c conda-forge pyarrow")
            st.code("Or ensure pip/setuptools/wheel are upgraded: python -m pip install --upgrade pip setuptools wheel")
            # show brief traceback for debugging
            st.text("Traceback (tail):")
            tb = traceback.format_exc().splitlines()[-5:]
            for line in tb:
                st.text(line)
            model = None
    else:
        st.warning(f"Model file not found at: {MODEL_FILE}")

    # Try to load npy data
    if os.path.exists(NPY_FILE):
        try:
            npy_data = np.load(NPY_FILE, allow_pickle=True)
        except Exception as e:
            st.error(f"Failed to load {NPY_FILE}: {e}")
            npy_data = None
    else:
        st.warning(f"NPY file not found at: {NPY_FILE}")

    # If model failed, return a DummyModel so the app remains usable
    if model is None:
        st.info("Using fallback DummyModel so predictions can still be demonstrated.")
        model = DummyModel()

    # If npy_data is None, provide an example array so UI doesn't break
    if npy_data is None:
        npy_data = np.arange(10)

    st.success("Assets load step completed (model may be a fallback).")
    return model, npy_data

# Load assets globally
model, npy_data = load_assets()

if model is not None:
    # Check if this is FAISS similarity search data
    if isinstance(model, dict) and 'faiss_index' in model and 'image_paths' in model:
        st.success("✅ Smart Closet FAISS Similarity Search System Loaded!")
        
        faiss_vectors = model['faiss_index']
        image_paths_data = model['image_paths']
        
        # Display dataset info
        st.subheader("📊 Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Items", len(image_paths_data))
        with col2:
            st.metric("Feature Dimension", faiss_vectors.shape[0] // len(image_paths_data) if len(image_paths_data) > 0 else 0)
        with col3:
            st.metric("Vector Data Size", f"{faiss_vectors.nbytes / (1024**2):.1f} MB")
        
        st.write("---")
        
        # Show sample image paths
        st.subheader("🔍 Find Similar Items in Your Closet")
        st.write("**Sample items in the database:**")
        sample_size = min(10, len(image_paths_data))
        for i in range(sample_size):
            st.text(f"{i}: {image_paths_data[i]}")
        
        st.write("---")
        
        # User input for similarity search
        st.subheader("Search for Similar Items")
        
        selected_index = st.number_input(
            f"Enter item index (0 to {len(image_paths_data) - 1})",
            min_value=0,
            max_value=len(image_paths_data) - 1,
            value=0,
            step=1
        )
        
        num_similar = st.slider("Number of similar items to find", min_value=1, max_value=20, value=5)
        
        if st.button("🔍 Find Similar Items"):
            try:
                st.write(f"**Selected Item (Index {selected_index}):**")
                selected_path = image_paths_data[selected_index]
                st.text(selected_path)
                
                # Extract just the filename for display
                filename = os.path.basename(selected_path)
                st.info(f"📁 Image: {filename}")
                
                # Display the selected image if it exists locally
                if os.path.exists(selected_path):
                    try:
                        st.image(selected_path, caption="Query Item", width=300)
                    except:
                        pass
                
                # Perform similarity search if FAISS is available
                if FAISS_AVAILABLE:
                    st.write("---")
                    st.subheader("🎯 Similar Items Found:")
                    
                    # Reshape faiss_vectors to proper 2D array
                    num_items = len(image_paths_data)
                    feature_dim = 2048  # Standard ResNet/image feature dimension
                    usable_size = num_items * feature_dim
                    
                    # Truncate to usable size in case of extra padding
                    vectors_truncated = faiss_vectors[:usable_size]
                    vectors_2d = vectors_truncated.reshape(num_items, feature_dim).astype('float32')
                    
                    # Build FAISS index
                    index = faiss.IndexFlatL2(feature_dim)
                    index.add(vectors_2d)
                    
                    # Get query vector
                    query_vector = vectors_2d[selected_index:selected_index+1]
                    
                    # Search for similar items (k+1 because first result is the query itself)
                    distances, indices = index.search(query_vector, num_similar + 1)
                    
                    # Display results (skip first one as it's the query itself)
                    for rank, (dist, idx) in enumerate(zip(distances[0][1:], indices[0][1:]), 1):
                        similarity_score = 1 / (1 + dist)
                        similar_path = image_paths_data[idx]
                        similar_filename = os.path.basename(similar_path)
                        
                        with st.container():
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric(f"#{rank}", f"{similarity_score:.3f}")
                            with col2:
                                st.write(f"**{similar_filename}**")
                                st.caption(f"Distance: {dist:.2f} | Index: {idx}")
                            
                            # Try to display image if exists
                            if os.path.exists(similar_path):
                                try:
                                    st.image(similar_path, width=200)
                                except:
                                    pass
                            st.write("")
                    
                    st.success(f"✅ Found {num_similar} similar items!")
                else:
                    st.warning("⚠️ FAISS library not available. Install it to enable similarity search:")
                    st.code("pip install faiss-cpu", language="bash")
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.text(traceback.format_exc())
    
    else:
        # Original prediction model interface
        st.subheader("Loaded Data Check")
        st.write(f"The first 5 elements of your loaded NPY file (`{NPY_FILE}`):")
        try:
            st.code(npy_data[:5])
        except Exception:
            st.code(str(npy_data))

        # --- Create Input Widgets for Prediction ---
        st.subheader("Make a Prediction")

        # Example Input 1 (REPLACE THIS WITH YOUR MODEL'S INPUT)
        input_feature_1 = st.slider("Feature A: Value between 0 and 10", 0.0, 10.0, 5.0)

        # Example Input 2 (REPLACE THIS WITH YOUR MODEL'S INPUT)
        input_feature_2 = st.number_input("Feature B: Enter a value", value=0.0, step=0.1)

        if st.button("Calculate Result"):
            # 1. Prepare data for the model
            input_array = np.array([[input_feature_1, input_feature_2]])

            # 2. Make Prediction
            try:
                if hasattr(model, 'predict'):
                    prediction = model.predict(input_array)
                    # 3. Display Result
                    st.metric("Predicted Output", value=f"{prediction[0]:.4f}")
                else:
                    st.error(f"The loaded model (type: {type(model)}) doesn't have a 'predict' method.")
            except Exception as e:
                st.error("Prediction failed. See error below.")
                st.text(str(e))
                tb = traceback.format_exc().splitlines()[-10:]
                for line in tb:
                    st.text(line)
# ...existing code...