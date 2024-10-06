import axios from 'axios';

// DOM 요소 가져오기
const inputElement = document.getElementById('question') as HTMLInputElement;
const submitButton = document.getElementById('submit') as HTMLButtonElement;
const responseContainer = document.getElementById('responses') as HTMLDivElement;

// 질문 제출 처리
submitButton.addEventListener('click', async () => {
    const question = inputElement.value;
    if (question) {
        try {
            const response = await axios.post('http://localhost:3000/api/question', { question });
            const answer = response.data.answer;
            const responseElement = document.createElement('div');
            responseElement.textContent = `질문자: ${question} - LLama: ${answer}`;
            responseContainer.appendChild(responseElement);
            inputElement.value = ''; // 입력 필드 초기화
        } catch (error) {
            console.error('Error:', error);
        }
    }
});
