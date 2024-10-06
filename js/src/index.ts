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
            const response = await axios.post('http://ec2-13-209-48-72.ap-northeast-2.compute.amazonaws.com:8000/api/question', { question });
            const answer = response.data.answer;
            const responseElement = document.createElement('div');
            responseElement.textContent = `Q: ${question} - A: ${answer}`;
            responseContainer.appendChild(responseElement);
            inputElement.value = ''; // 입력 필드 초기화
        } catch (error) {
            console.error('Error:', error);
        }
    }
});
