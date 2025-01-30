# LLM_Llama2_01
<br/>

## Prompt Engineering(프롬프트 엔지니어링)

#### Prompt(프롬프트)란?
생성형 AI가 특정 작업을 수행하도록 입력해주는 입력값.

#### Prompt Engineering(프롬프트 엔지니어링)이란?
생성형 AI가 고품질의, 최적의 결과물을 만들어낼 수 있도록 입력 프롬프트를 설계하고 구조화 하는 작업.

#### 좋은 결과물을 얻기 위한 6가지 전략
[https://platform.openai.com/docs/guides/prompt-engineering](https://platform.openai.com/docs/guides/prompt-engineering)
1. 지시사항을 명확하게 작성하기
    - 질문에 세부정보를 추가하여 질문에 연관성있는 답변 얻기
    - 모델에게 특정 인물이나 역할을 부여하기
    - 구분자(delimiters) 사용해서 입력된 문장들을 잘 구분해주기
    - 일을 완료하는데 필요한 단계를 구체적으로 알려주기
    - 예시를 제공하기
    - 원하는 출력값의 길이를 알려주기

2. 참고자료 제공하기 
    - 참고 자료를 바탕으로 답변하도록 하기
    - 참고 자료에서 인용해서 답변하도록 하기

3. 복잡한 작업을 단순한 작업들로 나누어주기
    - 질문의 의도를 나누어서, 사용자의 질문에 가장 적절한 지시사항을 알아볼 수 있도록 하기
    - 긴 대화가 필요한 애플리케이션에서는 대화내용을 요약하거나 이전 대화 내용을 필터링 하도록 하기
    - 긴 문서를 부분적으로 요약하고, 하나로 합쳐서 전체 요약을 만들도록 하기

4. 모델이 생각할 시간 주기
    - 결론을 서두르기 전에 자체적으로 해결책을 생각해보도록 하기
    - 모델의 사고 과정을 감추기 위해 독백(inner monologue)이나 여러 개의 연속된 질문을 하도록 하여 사용자에게 알아서 필요한 정보를 구분할 수 있게 하기
    - 이전 답변에서 빠진 것이 없는지 모델에게 다시 확인하기

5. 외부의 다른 도구들을 이용하기
    - 임베딩 기반 검색(embeddings-based search)을 사용하여 효과적으로 검색하도록 하기
    - 코드 실행을 사용하여 정확하게 계산하거나 외부 API 호출하기
    - 특정 기능에 대한 접근 권한을 모델에게 주기

6. 변겸사항을 체계적으로 테스트하기
    - 결과물을 gold-standard 정답과 비교하여 평가하기
<br/>

## Cloud 

#### Cloud란
인터넷을 통해 가상화된 컴퓨팅 자원과 서비스르 등을 제공받아 사용하는 기술. 사용자와 기업은 클라우드 컴퓨팅을 통해 직접 물리적 서버를 구축 및 관리없이 인터넷을 통해 서버, 저장공간, 데이터베이스, 네트워크, 소프트웨어 등 다양한 IT자원을 사용할 수 있다.

#### 클라우스 서비스 모델
1. IaaS (Infrastructure as a Service)
IaaS는 물리적인 컴퓨팅 자원을 가상화해서 제공하는 서비스
사용자가 가상 서버, 스토리지, 네트워크 등 필요한 인프라를 선택하여 사용할 수 있습니다.
사용자가 모든 설정을 직접하기 때문에 높은 유연성과 확장성을 제공하며, 사용자는 하드웨어 관리에서 벗어나 애플리케이션 개발이나 비즈니스에 집중할 수 있습니다.
예시: Amazon EC2, Microsoft Azure VM, Google Compute Engine

2. PaaS (Platform as a Service):
애플리케이션 개발을 위한 플랫폼을 가상화해서 제공하는 서비스
사용자는 인프라를 직접 관리할 필요 없이, 애플리케이션 코드 작성에만 집중할 수 있음.
데이터베이스, 미들웨어, 개발 도구 등 다양한 서비스와 기능을 통합하여 제공하며, 신속한 개발과 배포가 가능.
예시: Google App Engine, Heroku, PythonAnyWhere

3. SaaS (Software as a Service):
소프트웨어 애플리케이션을 인터넷 접속을 통해서 사용할 수 있도록 하는 서비스 
사용자는 소프트웨어를 설치하거나 유지 관리할 필요 없이 웹 브라우저를 통해 접근하고 사용 가능. 
모든 서비스를 맡기고 비지니스에 집중할 수 있음.
예시: Dropbox. Notion, Slack
<br/>
## On-premise

#### On-premise란?
클라우스 컴퓨팅 기술 이전의 전통적인 IT인프라 구축 방식으로, IT 시스템이나 소프트웨어를 기업 내부의 자체 서버에 전산 서버에 직접 설치하고 운영하는 방식. 클라우드 환경과는 달리, 모든 하드웨어와 소프트웨어를 기업 내부에서 직접 관리하여 유지 보수가 쉬우며, 보안상 관리가 수월하다.

#### 장단점
장점
1. 데이터 보호 및 보안이 수월하다
2. 기업이나 사용자에 맞게 커스텀하여 최적화된 환경을 구축할 수 있다.
3. 클라우드와 달리 외부 서비스 장애에 영향을 받지 않는다.

단점
1. 초기 인프라 구축에 비용이 많이 필요하다.
2. 유지 보수 및 관리에도 많은 비용이 들어간다.
3. 시스템을 확장하려면 추가적인 장비를 구입하는 등 빠르게 적용하기 쉽지 않다.