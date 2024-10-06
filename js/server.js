const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios'); // Axios 추가

const app = express();
const PORT = 3000;

app.use(bodyParser.json());
app.use(express.static('public'));

// 질문 API 엔드포인트
app.post('/api/question', async (req, res) => {
    const { question } = req.body;

    try {
        // FastAPI 서버로 질문 전송
        const response = await axios.post('https://ec2-13-209-48-72.ap-northeast-2.compute.amazonaws.com', { question });
        const answer = response.data.answer; // FastAPI의 응답에서 answer 추출

        // 클라이언트에 응답
        res.json({ answer });
    } catch (error) {
        console.error('FastAPI 요청 실패:', error);
        res.status(500).json({ answer: '서버 오류가 발생했습니다.' });
    }
});

app.listen(PORT, () => {
    console.log(`서버가 http://localhost:${PORT} 에서 실행 중입니다.`);
});
