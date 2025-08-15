import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input, SpatialDropout2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mediapipe as mp
import tensorflow as tf

# Configurações
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40
DATASET_PATH = "Samples"
TRAIN_DIR = os.path.join(DATASET_PATH, "Train")
VAL_DIR = os.path.join(DATASET_PATH, "Val")

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_region(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            hand_landmarks = results.multi_hand_landmarks[0]
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
            ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)
            margin = 20
            xmin = max(xmin - margin, 0)
            xmax = min(xmax + margin, w)
            ymin = max(ymin - margin, 0)
            ymax = min(ymax + margin, h)
            cropped = image[ymin:ymax, xmin:xmax]
            # Garantir 3 canais
            if len(cropped.shape) == 2 or cropped.shape[2] == 1:
                cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
            return cv2.resize(cropped, IMG_SIZE)
    # Caso não detecte mão, retorna imagem original em RGB
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.resize(image, IMG_SIZE)

# Geradores de dados
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED,
    subset='training',
    color_mode='rgb'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=SEED,
    subset='validation',
    color_mode='rgb'
)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    color_mode='rgb'
)

classes = list(train_generator.class_indices.keys())
num_classes = len(classes)

# Pesos de classe
labels = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

# Modelo EfficientNetB0 melhorado
def create_improved_model(fine_tune_at=170):
    input_tensor = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_tensor)
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    x = base_model.output
    x = SpatialDropout2D(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer=Adam(learning_rate=5e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

## Callbacks
callbacks = [
    ModelCheckpoint('modelLinux.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=1e-6)
]

# Treinamento
model = create_improved_model()
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Avaliação
best_model = tf.keras.models.load_model('modelLinux.h5')
loss, acc = best_model.evaluate(test_generator)
print(f"Acurácia final: {acc:.4f}")


# Grad-CAM
def generate_gradcam(model, img_array, class_index, layer_name='top_conv'):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)
    return heatmap

def display_gradcam(img_array, heatmap, alpha=0.4):
    img = img_array[0]
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.uint8(255 * img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + img
    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed_img.astype(np.uint8))
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()

# Exemplo Grad-CAM
sample_img, _ = test_generator[0]
img_array = np.expand_dims(sample_img[0], axis=0)
pred_class = np.argmax(best_model.predict(img_array))
heatmap = generate_gradcam(best_model, img_array, class_index=pred_class, layer_name='top_conv')
display_gradcam(img_array, heatmap)
