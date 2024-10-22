import matplotlib.pyplot as plt

# 손실 그래프
plt.figure(figsize=(12, 6))
plt.plot(logging_callback.losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 정확도 그래프 (정확도가 기록된 경우)
if logging_callback.accuracies:
    plt.figure(figsize=(12, 6))
    plt.plot(logging_callback.accuracies, label='Accuracy', color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
