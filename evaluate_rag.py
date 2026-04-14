"""
Скрипт для оценки качества RAG-системы через RAGAS
"""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from rag_assistant import ask_assistant
import config

# Пары (вопрос, эталонный ответ) для метрик с reference. Эталоны — по data/doc1.txt и doc2.txt;
# для вопроса вне базы — ожидаемая честная формулировка.
EVALUATION_DATA: list[tuple[str, str]] = [
    (
        "В течение скольких минут после начала обращения клиент должен получить ответ в рабочее время?",
        "Время ответа на обращение не должно превышать 2 минут в рабочее время.",
    ),
    (
        "Сколько одновременных сессий максимально разрешено на базовом тарифе?",
        "Максимум три одновременных сессии для базового тарифа.",
    ),
    (
        "Забыл пароль и не могу войти в учётную запись — какие шаги для восстановления доступа?",
        "На странице входа нажать «Забыли пароль?», ввести email от регистрации, получить письмо с инструкциями; если письма нет 10 минут — проверить «Спам» или написать в поддержку.",
    ),
    (
        "Где в приложении получить выгрузку данных и в каких форматах её можно скачать?",
        "Раздел «Настройки» → «Экспорт данных»; форматы CSV, JSON или PDF.",
    ),
    (
        "Какой юридический адрес головного офиса компании?",
        "В тексте базы знаний юридический адрес головного офиса не указан.",
    ),
]

EVALUATION_QUESTIONS = [q for q, _ in EVALUATION_DATA]
EVALUATION_REFERENCES = [a for _, a in EVALUATION_DATA]


def prepare_dataset(questions: list[str], reference_answers: list[str]) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов и эталонных ответов.

    Args:
        questions: список вопросов
        reference_answers: эталон для ground_truth (нужен для context_precision с reference)

    Returns:
        Dataset для RAGAS
    """
    if len(questions) != len(reference_answers):
        raise ValueError("Число вопросов и эталонных ответов должно совпадать")

    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []

    print("Получение ответов от ассистента...")

    for i, (question, reference) in enumerate(zip(questions, reference_answers), 1):
        print(f"  Обработка вопроса {i}/{len(questions)}: {question}")

        result = ask_assistant(question)

        questions_list.append(question)
        answers_list.append(result["answer"])
        context_texts = [chunk["document"] for chunk in result["context"]]
        contexts_list.append(context_texts)
        ground_truths_list.append(reference)

    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list,
    }

    return Dataset.from_dict(dataset_dict)


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы
    """
    print("=" * 60)
    print("Оценка качества RAG-системы через RAGAS")
    print("=" * 60)
    
    dataset = prepare_dataset(EVALUATION_QUESTIONS, EVALUATION_REFERENCES)

    print("\nЗапуск оценки метрик...")
    print(
        "Метрики: faithfulness, answer_relevancy (эвристика; для RU может быть 0), "
        "answer_similarity (ответ vs эталон, эмбеддинги), context_precision (контекст vs эталон), "
        "context_utilization (контекст vs ответ модели)"
    )
    
    # Убеждаемся, что переменная окружения установлена
    import os
    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
    
    # Настройка эмбеддингов и LLM для RAGAS
    # RAGAS требует использовать обертки для langchain объектов
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        
        # Создаём langchain объекты
        langchain_embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        langchain_llm = ChatOpenAI(
            model_name=config.CHAT_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0
        )
        
        # Обёртываем для RAGAS
        ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
        ragas_llm = LangchainLLMWrapper(langchain_llm)
        
        from ragas.metrics import (
            AnswerSimilarity,
            AnswerRelevancy,
            ContextPrecision,
            ContextUtilization,
            Faithfulness,
        )

        faithfulness_metric = Faithfulness(llm=ragas_llm)
        answer_relevancy_metric = AnswerRelevancy(
            llm=ragas_llm, embeddings=ragas_embeddings, strictness=2
        )
        # Сходство ответа RAG с эталоном (устойчивее, чем answer_relevancy для RU и коротких фактов)
        answer_similarity_metric = AnswerSimilarity(embeddings=ragas_embeddings)

        try:
            context_precision_metric = ContextPrecision(llm=ragas_llm, embeddings=ragas_embeddings)
        except TypeError:
            context_precision_metric = ContextPrecision(llm=ragas_llm)

        context_utilization_metric = ContextUtilization(llm=ragas_llm)

        metrics_to_use = [
            faithfulness_metric,
            answer_relevancy_metric,
            answer_similarity_metric,
            context_precision_metric,
            context_utilization_metric,
        ]

    except ImportError:
        print("Обёртки RAGAS недоступны, используем встроенные метрики с переменными окружения")
        from ragas.metrics._answer_similarity import answer_similarity
        from ragas.metrics._context_precision import context_utilization

        metrics_to_use = [
            faithfulness,
            answer_relevancy,
            answer_similarity,
            context_precision,
            context_utilization,
        ]
    except Exception as e:
        print(f"Используем встроенные метрики (предупреждение: {e})")
        from ragas.metrics._answer_similarity import answer_similarity
        from ragas.metrics._context_precision import context_utilization

        metrics_to_use = [
            faithfulness,
            answer_relevancy,
            answer_similarity,
            context_precision,
            context_utilization,
        ]
    
    # Запускаем оценку
    result = evaluate(
        dataset=dataset,
        metrics=metrics_to_use
    )
    
    # Выводим результаты
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 60)
    
    # RAGAS возвращает Dataset с результатами
    # Вычисляем средние значения метрик, игнорируя nan
    import math
    
    faithfulness_values = [v for v in result['faithfulness'] if not math.isnan(v)] if result['faithfulness'] else []
    answer_relevancy_values = [v for v in result['answer_relevancy'] if not math.isnan(v)] if result['answer_relevancy'] else []
    try:
        asim_col = result["answer_similarity"]
    except KeyError:
        asim_col = []
    answer_similarity_values = [v for v in asim_col if not math.isnan(v)] if asim_col else []
    context_precision_values = [v for v in result['context_precision'] if not math.isnan(v)] if result['context_precision'] else []
    # evaluate() возвращает EvaluationResult (не HuggingFace Dataset): список метрик — в _scores_dict
    try:
        cu_col = result["context_utilization"]
    except KeyError:
        cu_col = []
    context_utilization_values = [v for v in cu_col if not math.isnan(v)] if cu_col else []

    avg_faithfulness = sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0
    avg_answer_relevancy = sum(answer_relevancy_values) / len(answer_relevancy_values) if answer_relevancy_values else float('nan')
    avg_answer_similarity = (
        sum(answer_similarity_values) / len(answer_similarity_values)
        if answer_similarity_values
        else float('nan')
    )
    avg_context_precision = sum(context_precision_values) / len(context_precision_values) if context_precision_values else 0
    avg_context_utilization = (
        sum(context_utilization_values) / len(context_utilization_values)
        if context_utilization_values
        else float('nan')
    )

    print(f"\nFaithfulness (верность ответа): {avg_faithfulness:.4f}")
    if not math.isnan(avg_answer_relevancy):
        print(f"Answer Relevancy (эвристика по вопросу↔ответ): {avg_answer_relevancy:.4f}")
    else:
        print("Answer Relevancy: не удалось вычислить")
    if not math.isnan(avg_answer_similarity):
        print(f"Answer Similarity (ответ vs эталон, семантика): {avg_answer_similarity:.4f}")
    print(f"Context Precision (контекст vs эталонный ответ): {avg_context_precision:.4f}")
    if not math.isnan(avg_context_utilization):
        print(f"Context Utilization (контекст vs ответ RAG): {avg_context_utilization:.4f}")
    
    # Выводим детали по каждому вопросу
    print("\n" + "=" * 60)
    print("ДЕТАЛИ ПО ВОПРОСАМ")
    print("=" * 60)
    
    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\nВопрос {i+1}: {question}")
        print(f"  Faithfulness: {result['faithfulness'][i]:.4f} ---точность ответа")
        ar_val = result['answer_relevancy'][i]
        if not math.isnan(ar_val):
            print(f"  Answer Relevancy: {ar_val:.4f} ---релевантность ответа вопросу")
        else:
            print(f"  Answer Relevancy: не удалось вычислить ---релевантность ответа вопросу")
        if asim_col:
            print(f"  Answer Similarity: {asim_col[i]:.4f} ---ответ vs эталон (эмбеддинги)")
        print(f"  Context Precision: {result['context_precision'][i]:.4f} ---контекст vs эталон")
        if cu_col:
            print(f"  Context Utilization: {cu_col[i]:.4f} ---контекст vs ответ модели")

    print("\n" + "=" * 60)
    print("Оценка завершена!")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_rag_system()

