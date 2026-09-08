# HDAT-DS Python 기초 10강 — 파이썬을 처음 배우는 사람을 위한 교재

> 대상: `print("안녕하세요")` 정도만 본 적 있고, `pd`, `np`, `DataFrame`, `shape`가 아직 낯선 학습자  
> 목표: 이 10강을 끝낸 뒤 기존 **HDAT-DS 1~34강**을 무리 없이 시작한다.  
> 원칙: 모든 예제는 외부 데이터 없이 실행한다. 딥러닝 프레임워크는 **PyTorch만** 사용한다.

---

# 입문 1강. 파이썬 실행 환경과 파일, Jupyter Notebook

## 이 과정은 어떻게 공부하나요?

처음부터 코드를 외우려고 하지 않아도 된다. 코드가 하는 일을 한 줄씩 말로 설명할 수 있게 되는 것이 먼저다. 매 강의에서 다음 순서로 공부한다.

1. `왜 필요한가`를 읽어 오늘 배울 도구의 용도를 파악한다.
2. 예제 코드를 **복사만 하지 말고 직접 입력**한다.
3. 실행 전 결과를 한 번 예상하고, 실행 후 실제 결과와 비교한다.
4. 변수의 `type`, 배열·표·텐서의 `shape`, 숫자의 `dtype`을 습관처럼 확인한다.
5. 실습문제를 먼저 풀고 나서 접힌 정답을 연다.
6. 완료 기준을 모두 만족할 때 다음 강으로 간다.

권장 시간은 강의당 90–150분이다. 한 번에 모두 이해하지 못해도 괜찮다. 특히 5–10강은 하루에 한 강씩 공부하고 다음 날 같은 코드를 빈 화면에서 다시 써 본다.

### 코드를 어디에서 실행하나요?

가장 쉬운 환경은 Jupyter Notebook이다. 한 칸을 <strong>셀(cell)</strong>이라고 부르며, 셀 안에 코드를 넣고 `Shift + Enter`를 누르면 실행된다. `.py` 파일로 공부해도 된다. 이 교재의 Python 코드 블록은 필요한 라이브러리를 불러오는 명령과 예제 데이터를 각각 포함하므로, 원칙적으로 <strong>새 노트북의 빈 셀 하나에 코드 블록 전체를 붙여 넣어 독립 실행</strong>할 수 있다.

#### 처음 한 번만: Python이 실행되는지 확인하기

아래 명령은 Python 코드 셀이 아니라 **터미널**에 입력한다. Mac에서는 `Command + Space`를 누르고 `터미널`을 검색한다. Windows에서는 시작 메뉴에서 `PowerShell`을 검색한다. 검은색이나 파란색 창이 열려도 정상이다.

먼저 다음 한 줄을 입력하고 `Enter`를 누른다.

```bash
python --version
```

`Python 3.x.x`처럼 나오면 앞으로 `python`이라고 적힌 명령을 그대로 사용한다. `command not found`, `'python' is not recognized`처럼 나오면 다음 후보를 **한 줄씩** 시도한다.

```bash
python3 --version
```

```bash
py --version
```

성공한 명령을 계속 사용한다. 예를 들어 `python3 --version`만 성공했다면 이후 `python -m ...` 대신 `python3 -m ...`라고 입력한다. 세 명령이 모두 실패하면 Python 3가 아직 없거나 터미널이 설치 위치를 찾지 못한 상태다. [Python 공식 다운로드](https://www.python.org/downloads/)에서 Python 3를 설치한 뒤 터미널을 완전히 닫았다가 다시 열고 같은 확인을 반복한다.

#### 필요한 패키지와 Jupyter 설치하기

위에서 `python`이 성공했다면 터미널에 다음을 한 번 실행한다.

```bash
python -m pip install numpy pandas matplotlib torch jupyter
```

`python3`가 성공했다면 첫 단어만 다음처럼 바꾼다.

```bash
python3 -m pip install numpy pandas matplotlib torch jupyter
```

Windows에서 `py`만 성공했다면 다음을 사용한다.

```bash
py -m pip install numpy pandas matplotlib torch jupyter
```

화면에 여러 줄이 지나가고 마지막에 `Successfully installed ...` 또는 `Requirement already satisfied ...`가 나오면 준비가 끝난 것이다. 권한 오류가 나면 관리자 명령부터 시도하지 말고, 오류의 마지막 3~5줄을 복사해 검색하거나 도움을 요청한다. PyTorch 설치만 실패한다면 운영체제와 장치에 맞는 명령을 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 확인한다.

#### Jupyter 열기 → 새 노트북 만들기 → 코드 실행하기

설치할 때 성공한 Python 명령에 맞춰 터미널에서 **셋 중 하나만** 실행한다.

```bash
python -m jupyter notebook
```

```bash
python3 -m jupyter notebook
```

```bash
py -m jupyter notebook
```

그다음 순서는 다음과 같다.

1. 잠시 기다리면 웹 브라우저에 Jupyter 파일 목록이 열린다. 터미널 창은 Jupyter가 실행되는 동안 닫지 않는다.
2. 새 파일은 화면의 `New` 또는 `+` 버튼에서 `Python 3 (ipykernel)`을 선택해 만든다. 메뉴 이름은 버전에 따라 조금 다를 수 있다.
3. 빈 셀에 `print("준비 완료")`를 입력하고 `Shift + Enter`를 누른다.
4. 셀 아래에 `준비 완료`가 보이면 첫 실행에 성공한 것이다.
5. 파일 이름을 눌러 `python-basics.ipynb`처럼 알아볼 수 있는 이름으로 바꾸고 `Ctrl/Cmd + S`로 저장한다.
6. 마지막 점검에서는 `Kernel` 메뉴의 **Restart and Run All**에 해당하는 항목을 눌러 첫 셀부터 전부 다시 실행한다.

다운로드한 `.ipynb` 파일을 열 때는 Jupyter 파일 목록에서 그 파일이 있는 폴더로 이동해 파일 이름을 누른다. 파일이 보이지 않으면 가장 쉬운 방법은 Jupyter 화면에서 제공하는 업로드 버튼으로 `.ipynb` 파일을 올리는 것이다. 이 사이트의 `python-data-basics-starter.ipynb`는 빈칸을 직접 채우는 연습용이고, `python-data-basics-solution.ipynb`는 막힌 뒤 비교하는 정답용이다.

> **지금 설치가 어려워도 괜찮다.** 우선 1강을 읽으며 코드·터미널·셀의 차이를 익힌 뒤 다시 이 절로 돌아와도 된다. 설치는 개인 공부용 컴퓨터에서만 한다. 시험 환경에는 정해진 Python·라이브러리 버전과 사용 규칙이 있으므로 시험 직전 공식 안내를 다시 확인한다.

### 앞으로 계속 보게 될 네 가지 객체

| 이름 | 쉬운 설명 | 예시 | 가장 먼저 확인할 것 |
|---|---|---|---|
| Python `list` | 여러 값을 순서대로 담는 자료구조 | `[10, 20, 30]` | `len(x)`, `type(x)` |
| NumPy `ndarray` | 같은 종류의 숫자를 빠르게 계산하는 배열 | `np.array([10, 20])` | `x.shape`, `x.dtype` |
| pandas `DataFrame` | 열 이름이 있는 2차원 표 | 키, 나이, 점수 표 | `x.shape`, `x.dtypes`, `x.head()` |
| PyTorch `Tensor` | 모델 계산과 미분에 쓰는 다차원 숫자 배열 | `torch.tensor(...)` | `x.shape`, `x.dtype`, `x.device` |

## 1. 왜 필요한가

코드가 틀리지 않았는데도 "파일을 찾을 수 없습니다", "모듈이 없습니다", "아까 만든 변수가 없습니다" 같은 오류가 날 수 있다. 대개 **어떤 Python을 실행 중인지**, **현재 어느 폴더를 보고 있는지**, **노트북 셀을 어떤 순서로 실행했는지**가 원인이다. 데이터 분석을 시작하기 전에 실행 환경을 확인하는 방법부터 알아야 이후 오류를 스스로 해결할 수 있다.

이 강의를 마치면 다음을 할 수 있다.

- Python 코드, 노트북 셀, 터미널 명령의 차이를 설명한다.
- 셀을 실행하고 커널을 재시작한 뒤 위에서 아래로 다시 실행한다.
- 현재 작업 폴더와 파일 경로를 `pathlib`로 확인한다.
- 파일을 만들고 읽을 때 문자열 인코딩 `UTF-8`을 지정한다.
- 오류 메시지의 마지막 줄에서 오류 종류와 원인을 찾는다.

## 2. 프로그램·인터프리터·패키지

Python은 사람이 쓴 코드를 컴퓨터가 실행하도록 해 주는 프로그래밍 언어다. Python **인터프리터**는 그 코드를 실제로 읽고 실행하는 프로그램이다. NumPy, pandas, Matplotlib, PyTorch는 Python에 기능을 추가하는 <strong>패키지(라이브러리)</strong>다.

- Python 코드: `score = 80`, `print(score)`처럼 Python 문법으로 쓴 명령
- 터미널 명령: `python --version`, `python -m pip install ...`처럼 운영체제에 내리는 명령
- 노트북: 코드와 설명을 셀 단위로 섞어 기록하는 문서
- 커널(kernel): 현재 노트북의 코드를 실행하고 변수를 기억하는 Python 프로세스

터미널 명령을 Python 셀에 그대로 넣으면 `SyntaxError`가 날 수 있다. 반대로 `score = 80`을 운영체제 터미널에 바로 입력하면 셸이 이해하지 못한다. "지금 입력하는 칸이 Python 셀인가, 터미널인가"를 먼저 확인한다.

## 3. 첫 코드와 실행 순서

아래 코드 전체를 새 셀에 입력하고 `Shift + Enter`를 누른다.

```python
message = "HDAT 공부를 시작합니다"
lesson = 1

print(message)
print("현재 강의:", lesson)
print("코드는 위에서 아래로 실행됩니다")
```

예상 출력:

```text
HDAT 공부를 시작합니다
현재 강의: 1
코드는 위에서 아래로 실행됩니다
```

`#` 뒤의 문장은 주석이다. Python이 실행하지 않으며 사람에게 뜻을 설명한다.

```python
# 이 줄은 실행되지 않습니다.
temperature = 24.5  # 등호 오른쪽 값을 왼쪽 이름에 저장합니다.
print(temperature)
```

예상 출력:

```text
24.5
```

### 노트북의 중요한 함정: 화면 순서와 실행 순서는 다를 수 있다

노트북은 실행한 셀의 변수를 커널 메모리에 남긴다. 아래쪽 셀을 먼저 실행하면 화면상 위에 코드가 있어도 변수가 아직 만들어지지 않았다. 반대로 이미 지운 셀에서 만든 변수가 메모리에 남아 코드가 우연히 작동할 수도 있다.

안전한 점검 순서:

1. 파일을 저장한다(`Ctrl/Cmd + S`).
2. 커널을 재시작한다.
3. 첫 셀부터 마지막 셀까지 위에서 아래로 모두 실행한다.
4. 같은 결과가 나오는지 확인한다.

## 4. 현재 Python과 설치된 패키지 확인

```python
import sys

print("Python 실행 파일:", sys.executable)
print("Python 버전:", sys.version_info[:3])
```

예상 출력 형식은 다음과 같다. 경로와 버전 숫자는 컴퓨터마다 다르다.

```text
Python 실행 파일: /.../python
Python 버전: (3, 11, 9)
```

패키지를 가져오는 명령이 `import`다. 관례적으로 긴 이름을 짧게 부른다.

```python
import numpy as np
import pandas as pd
import matplotlib
import torch

print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("PyTorch:", torch.__version__)
```

예상 출력은 설치된 버전에 따라 숫자가 달라진다.

```text
NumPy: ...
pandas: ...
Matplotlib: ...
PyTorch: ...
```

`np`와 `pd`는 별도 패키지가 아니다. 각각 `numpy`, `pandas`에 붙이는 널리 쓰이는 짧은 별명이다. `import pandas as pd`를 실행한 뒤에만 `pd.DataFrame(...)`을 쓸 수 있다.

## 5. 폴더와 경로 이해하기

파일 경로는 컴퓨터 안에서 파일의 주소다. 상대 경로 `data/train.csv`는 **현재 작업 폴더**를 기준으로 찾는다. 초심자에게는 문자열을 직접 이어 붙이기보다 표준 라이브러리 `pathlib.Path`가 안전하다.

```python
from pathlib import Path

work_dir = Path.cwd()
print("현재 작업 폴더:", work_dir)
print("폴더인가요?:", work_dir.is_dir())
```

예상 출력 형식:

```text
현재 작업 폴더: /사용자마다/다른/경로
폴더인가요?: True
```

**선택 학습 — 지금은 실행 결과만 확인해도 된다.** 아래 파일 쓰기와 `with` 문법은 함수·들여쓰기 학습 후 돌아와도 된다. 1강에서는 실행 위치와 출력만 확인하면 충분하다. 외부 파일 없이 임시 폴더를 사용하며, `with` 블록이 끝나면 임시 파일은 자동 정리된다.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as temp_name:
    folder = Path(temp_name)
    file_path = folder / "memo.txt"

    file_path.write_text("첫 번째 데이터 파일\nUTF-8 한글", encoding="utf-8")
    loaded = file_path.read_text(encoding="utf-8")

    print("파일 존재:", file_path.exists())
    print("파일 이름:", file_path.name)
    print(loaded)
```

예상 출력:

```text
파일 존재: True
파일 이름: memo.txt
첫 번째 데이터 파일
UTF-8 한글
```

`folder / "memo.txt"`의 `/`는 나눗셈이 아니라 `Path` 객체에서 경로를 연결하는 연산이다. Windows와 macOS/Linux의 경로 표기 차이를 `pathlib`가 처리한다.

## 6. 오류 메시지를 읽는 가장 간단한 법

오류 메시지가 길어도 다음 순서로 읽으면 원인을 찾기 쉽다.

1. 맨 마지막 줄의 오류 이름: `NameError`, `FileNotFoundError`, `ModuleNotFoundError` 등
2. 마지막 줄의 설명: 어떤 이름·파일·모듈이 문제인지
3. 그 위의 `line ...`: 내 코드의 어느 줄인지
4. 오류가 난 줄 바로 앞에서 변수의 `type`, 값, 경로를 출력

**선택 미리보기:** 아래 `if`/`else`는 입문 4강에서 배운다. 지금은 오류 표만 읽고 다음 절로 넘어가도 된다. 아래 코드는 파일 존재 여부를 먼저 확인한다.

```python
from pathlib import Path

file_path = Path("definitely_missing_file.csv")

if file_path.exists():
    text = file_path.read_text(encoding="utf-8")
    print(text)
else:
    print("파일이 없습니다:", file_path.resolve())
    print("현재 폴더와 파일명을 먼저 확인하세요.")
```

예상 출력 형식:

```text
파일이 없습니다: /.../definitely_missing_file.csv
현재 폴더와 파일명을 먼저 확인하세요.
```

## 7. 흔한 오류와 해결법

| 증상 | 뜻 | 먼저 할 일 |
|---|---|---|
| `NameError: name 'x' is not defined` | `x`를 만들기 전에 사용함 | 변수를 만드는 셀부터 다시 실행 |
| `ModuleNotFoundError: No module named ...` | 현재 Python 환경에 패키지가 없음 | `sys.executable` 확인 후 같은 환경에 설치 |
| `FileNotFoundError` | 현재 작업 폴더 기준으로 파일이 없음 | `Path.cwd()`, `path.resolve()`, `path.exists()` 출력 |
| `SyntaxError` | Python 문법으로 해석할 수 없음 | 괄호·따옴표·콜론, 터미널 명령 혼입 확인 |
| 한글이 깨짐 | 읽기/쓰기 인코딩 불일치 가능 | 가능하면 양쪽에 `encoding="utf-8"` 지정 |
| 위 셀을 지웠는데 코드가 됨 | 커널에 이전 변수가 남음 | 커널 재시작 후 전체 실행 |

## 8. 실습문제

### 오늘의 필수 실습 3개

1. Python 셀에서 `print("첫 실행")`을 실행하라. 괄호 안 문장을 자신의 말로 바꾸고 다시 실행하라.
2. `print(2 + 3)`을 실행하라. 예상값 5와 비교한 다음 3을 7로 바꾸어 결과를 예상하고 확인하라.
3. 커널을 재시작한 뒤 위 두 셀을 위에서 아래로 실행하라. `python --version`은 Python 코드 셀이 아니라 터미널에 입력하는 명령이라고 설명하라.

<details><summary>필수 실습 확인</summary>

첫 코드는 괄호 안 문자열을 출력한다. 덧셈 결과는 처음 5, 수정 후 9다. 커널은 Python 코드의 실행 상태를 기억하는 공간이다. 재시작 후에는 필요한 셀을 다시 실행한다. `print(...)`는 Python 코드이고 `python --version`은 터미널 명령이다.

</details>

### 선택 확장 실습 — 입문 2~4강 후 돌아오기

다음은 변수·튜플·조건문을 배운 뒤에 풀어 볼 추가 문제다. 1강 완료 조건이 아니다. 정답을 열기 전에 새 셀에서 직접 해결한다.

1. `"Python 기초"`와 숫자 `10`을 각각 변수에 저장하고 한 줄에 출력하라.
2. 현재 실행 중인 Python의 주 버전·부 버전을 튜플 형태로 출력하라.
3. 현재 작업 폴더 아래에 있다고 가정한 `data/train.csv` 경로를 `Path`로 만들고 출력하라. 실제 파일은 만들지 않는다.
4. 임시 폴더에 한글 두 줄을 저장하고 다시 읽어 두 줄이 같은지 확인하라.
5. `numpy`를 `np`라는 별명으로 가져오고 버전을 출력하라.
6. `unknown.csv`가 없을 때 오류를 내지 않고 `"파일 확인 필요"`를 출력하라.

<details>
<summary>입문 1강 정답·해설 펼치기</summary>

### 1번

```python
course = "Python 기초"
count = 10
print(course, count)
```

예상 출력: `Python 기초 10`. `print`에 값을 쉼표로 나누어 주면 기본적으로 사이에 공백을 넣는다.

### 2번

```python
import sys

version = (sys.version_info.major, sys.version_info.minor)
print(version)
```

예상 출력 형식: `(3, 11)`. 설치 환경에 따라 두 번째 숫자는 다를 수 있다.

### 3번

```python
from pathlib import Path

path = Path.cwd() / "data" / "train.csv"
print(path)
```

예상 출력은 현재 폴더로 시작하고 `data/train.csv`로 끝난다. 아직 파일 존재 여부를 묻지 않았으므로 없어도 오류가 아니다.

### 4번

```python
from pathlib import Path
from tempfile import TemporaryDirectory

original = "첫 줄\n둘째 줄"
with TemporaryDirectory() as temp_name:
    path = Path(temp_name) / "korean.txt"
    path.write_text(original, encoding="utf-8")
    restored = path.read_text(encoding="utf-8")
    print(restored)
    print("같음:", restored == original)
```

예상 출력의 마지막 줄은 `같음: True`다.

### 5번

```python
import numpy as np

print(np.__version__)
```

숫자는 환경마다 다르다. `np`는 관례적인 별명이다.

### 6번

```python
from pathlib import Path

path = Path("unknown.csv")
if not path.exists():
    print("파일 확인 필요")
else:
    print(path.read_text(encoding="utf-8"))
```

예상 출력: `파일 확인 필요`. 존재 여부를 먼저 확인했기 때문에 `FileNotFoundError`가 나지 않는다.

</details>

## 완료 기준

- [ ] Python 셀과 터미널 명령의 차이를 말할 수 있다.
- [ ] 커널을 재시작하고 노트북 전체를 위에서 아래로 실행했다.
- [ ] `sys.executable`은 실행 환경, `Path.cwd()`는 작업 폴더를 알려 준다는 것을 확인했다.
- [ ] `import numpy as np`, `import pandas as pd`에서 `np`, `pd`가 별명임을 안다.
- [ ] 오늘의 필수 실습 3개를 직접 실행하고 결과를 설명했다. 선택 확장은 건너뛰어도 된다.

---

# 입문 2강. 변수·자료형·연산자

## 1. 왜 필요한가

데이터 분석 코드는 결국 값을 이름에 저장하고, 자료형에 맞는 연산을 수행하는 과정이다. 숫자처럼 보이는 문자열 `"100"`과 숫자 `100`은 다르다. 이 차이를 모르면 합계가 이상해지거나 모델 입력을 만들 때 오류가 난다.

이 강의를 마치면 다음을 할 수 있다.

- 변수와 대입의 뜻을 설명한다.
- `int`, `float`, `str`, `bool`, `None`을 구분한다.
- 사칙연산, 비교연산, 논리연산을 사용한다.
- 문자열을 숫자로 변환하고 f-string으로 결과를 표현한다.
- `=`와 `==`, `/`와 `//`, `and`와 `&`의 용도가 다름을 안다.

## 2. 변수는 값에 붙이는 이름

`=`는 "같다"가 아니라 오른쪽 값을 왼쪽 이름에 **대입한다**는 뜻이다.

```python
age = 29
height = 172.5
name = "민수"
is_beginner = True
missing_value = None

print(age, type(age))
print(height, type(height))
print(name, type(name))
print(is_beginner, type(is_beginner))
print(missing_value, type(missing_value))
```

예상 출력:

```text
29 <class 'int'>
172.5 <class 'float'>
민수 <class 'str'>
True <class 'bool'>
None <class 'NoneType'>
```

| 자료형 | 뜻 | 예시 | 주의 |
|---|---|---|---|
| `int` | 정수 | `3`, `-10`, `0` | 소수점 없음 |
| `float` | 실수 | `3.14`, `-0.5` | 컴퓨터 근사값이므로 아주 작은 오차 가능 |
| `str` | 문자열 | `"3.14"`, `"서울"` | 따옴표 속 숫자도 문자열 |
| `bool` | 참/거짓 | `True`, `False` | 첫 글자가 대문자 |
| `None` | 아직 값이 없음 | `result = None` | 문자열 `"None"`과 다름 |

변수 이름은 의미 있게 쓴다. `x1`보다 `train_score`, `a`보다 `sample_count`가 나중에 읽기 쉽다. 공백 대신 밑줄을 사용하고 숫자로 시작하지 않는다.

## 3. 숫자 연산

```python
a = 17
b = 5

print("더하기:", a + b)
print("빼기:", a - b)
print("곱하기:", a * b)
print("나누기:", a / b)
print("몫:", a // b)
print("나머지:", a % b)
print("거듭제곱:", a ** 2)
```

예상 출력:

```text
더하기: 22
빼기: 12
곱하기: 85
나누기: 3.4
몫: 3
나머지: 2
거듭제곱: 289
```

`/`는 일반 나눗셈이라 결과가 `float`다. `//`는 내림 나눗셈이고 `%`는 나머지다. 우선순위가 헷갈리면 괄호를 쓴다.

```python
score1 = 80
score2 = 90
score3 = 70

mean_score = (score1 + score2 + score3) / 3
print(mean_score)
print(round(mean_score, 1))
```

예상 출력:

```text
80.0
80.0
```

## 4. 문자열과 형 변환

문자열은 `+`로 이어 붙일 수 있지만 문자열과 숫자를 바로 더할 수는 없다. 출력에는 f-string이 읽기 쉽다.

```python
name = "지우"
score = 87.5

sentence = f"{name}님의 점수는 {score:.1f}점입니다."
print(sentence)
print("이름 길이:", len(name))
print("대문자:", "python".upper())
```

예상 출력:

```text
지우님의 점수는 87.5점입니다.
이름 길이: 2
대문자: PYTHON
```

사용자 입력이나 CSV에서 숫자가 문자열로 들어오는 일이 많다. 계산 전에 변환한다.

```python
raw_count = "12"
raw_rate = "0.75"

count = int(raw_count)
rate = float(raw_rate)

print(count + 3)
print(rate * 100)
print(type(count), type(rate))
```

예상 출력:

```text
15
75.0
<class 'int'> <class 'float'>
```

`int("12.5")`는 바로 변환되지 않는다. 소수 문자열이라면 먼저 `float("12.5")`로 읽고, 정말 소수 부분을 버리는 것이 의도일 때만 `int(...)`를 적용한다.

## 5. 비교와 논리 연산

비교 결과는 `bool`이다.

```python
score = 82
attendance = 0.9

print(score >= 80)
print(score == 82)
print(score != 100)
print((score >= 80) and (attendance >= 0.8))
print((score >= 90) or (attendance >= 0.8))
print(not (score < 60))
```

예상 출력:

```text
True
True
True
True
True
True
```

- `=`: 값을 저장
- `==`: 두 값이 같은지 비교
- `!=`: 다른지 비교
- `and`: 두 조건이 모두 참
- `or`: 둘 중 하나 이상 참
- `not`: 참과 거짓을 뒤집음

Python의 단일 `bool`에는 `and`, `or`, `not`을 쓴다. 나중에 NumPy·pandas의 여러 값을 한꺼번에 비교할 때는 괄호와 `&`, `|`, `~`를 사용한다. 둘을 섞어 쓰지 않는다.

## 6. 재대입과 누적

변수는 새 값으로 다시 연결할 수 있다.

```python
total = 0
print(total)

total = total + 10
print(total)

total += 5
print(total)
```

예상 출력:

```text
0
10
15
```

`total += 5`는 보통 `total = total + 5`를 간단히 쓴 것이다. 반복문에서 합계를 누적할 때 자주 사용한다.

## 7. 부동소수점은 근사값

```python
value = 0.1 + 0.2

print(value)
print(value == 0.3)
print(abs(value - 0.3) < 1e-9)
```

예상 출력:

```text
0.30000000000000004
False
True
```

컴퓨터의 `float`는 많은 실수를 이진수로 정확히 표현하지 못한다. 모델 평가값 같은 실수를 비교할 때 완전한 일치보다 허용 오차를 둘 수 있다. NumPy에서는 뒤에 배울 `np.isclose`를 많이 쓴다.

## 8. 흔한 오류와 해결법

| 실수 | 원인 | 해결 방법 |
|---|---|---|
| `"10" + 5`에서 `TypeError` | 문자열과 정수를 더함 | 계산 목적이면 `int("10") + 5` |
| `score = 80`을 비교식으로 착각 | `=`는 대입 | 비교는 `score == 80` |
| `int("3.5")`에서 `ValueError` | 정수 표기가 아닌 문자열 | `float("3.5")` 사용 |
| `true`, `false`에서 `NameError` | Python은 대소문자 구분 | `True`, `False` |
| `0.1 + 0.2 == 0.3`이 거짓 | 부동소수점 근사 | 허용 오차로 비교 |
| 문자열 숫자 합계가 `"1020"` | `+`가 문자열 연결을 수행 | 먼저 숫자로 변환 |

## 9. 실습문제

1. 섭씨 `25`도를 화씨로 바꾸라. 공식은 `섭씨 * 9 / 5 + 32`다.
2. `125`분을 `2시간 5분`처럼 몫과 나머지를 이용해 출력하라.
3. 문자열 `"42"`와 `"3.5"`를 숫자로 바꿔 합을 출력하라.
4. 점수 `78`, 출석률 `0.85`일 때 "점수 70 이상 **그리고** 출석률 0.8 이상"인지 출력하라.
5. 이름 `"하늘"`, 시도 횟수 `3`, 정확도 `0.87654`를 `하늘: 3회, 정확도 87.7%` 형식으로 출력하라.
6. `0.3 - 0.2`가 `0.1`에 매우 가까운지 오차 `1e-9` 기준으로 판단하라.

<details>
<summary>입문 2강 정답·해설 펼치기</summary>

### 1번

```python
celsius = 25
fahrenheit = celsius * 9 / 5 + 32
print(fahrenheit)
```

예상 출력: `77.0`. 곱셈과 나눗셈이 덧셈보다 먼저 계산된다.

### 2번

```python
minutes = 125
hours = minutes // 60
remaining = minutes % 60
print(f"{hours}시간 {remaining}분")
```

예상 출력: `2시간 5분`.

### 3번

```python
integer_text = "42"
float_text = "3.5"
result = int(integer_text) + float(float_text)
print(result)
```

예상 출력: `45.5`. 소수점이 있는 문자열은 `float`로 변환한다.

### 4번

```python
score = 78
attendance = 0.85
passed = (score >= 70) and (attendance >= 0.8)
print(passed)
```

예상 출력: `True`.

### 5번

```python
name = "하늘"
trials = 3
accuracy = 0.87654
print(f"{name}: {trials}회, 정확도 {accuracy:.1%}")
```

예상 출력: `하늘: 3회, 정확도 87.7%`. f-string의 `.1%`는 100을 곱한 백분율을 소수 첫째 자리로 표시한다.

### 6번

```python
value = 0.3 - 0.2
is_close = abs(value - 0.1) < 1e-9
print(value)
print(is_close)
```

예상 출력의 마지막 줄은 `True`다.

</details>

## 완료 기준

- [ ] `int`, `float`, `str`, `bool`, `None`의 예를 하나씩 만들 수 있다.
- [ ] `=`와 `==`, `/`와 `//`의 차이를 설명할 수 있다.
- [ ] 문자열 숫자를 계산 가능한 숫자로 변환할 수 있다.
- [ ] f-string으로 소수점과 백분율을 표현할 수 있다.
- [ ] 실습 6문제 중 5문제 이상을 정답 없이 풀었다.

---

# 입문 3강. 리스트·튜플·딕셔너리·집합과 인덱싱

## 1. 왜 필요한가

한 사람의 점수 하나만 다룰 때는 변수 하나면 된다. 하지만 샘플 1,000개의 점수나 여러 열 이름을 다루려면 여러 값을 묶는 방법이 필요하다. Python의 기본 자료구조를 이해하면 NumPy 배열, pandas 표, PyTorch 텐서의 indexing도 훨씬 쉽게 배울 수 있다.

이 강의를 마치면 다음을 할 수 있다.

- `list`, `tuple`, `dict`, `set`을 목적에 맞게 선택한다.
- 0부터 시작하는 index와 끝이 제외되는 slice를 사용한다.
- 가변(mutable)과 불변(immutable)의 차이를 설명한다.
- 얕은 복사와 같은 객체를 가리키는 대입을 구분한다.
- 중첩 자료구조에서 원하는 값을 꺼낸다.

## 2. list: 순서가 있고 바꿀 수 있는 값 묶음

리스트는 대괄호 `[]`로 만든다. 값의 **순서**가 유지되고, 값을 추가·수정·삭제할 수 있다.

```python
scores = [80, 95, 72]

print(scores)
print(type(scores))
print("개수:", len(scores))
print("첫 값:", scores[0])
print("마지막 값:", scores[-1])
```

예상 출력:

```text
[80, 95, 72]
<class 'list'>
개수: 3
첫 값: 80
마지막 값: 72
```

Python의 index는 `0`부터 시작한다. 원소가 3개라면 유효한 양수 index는 `0`, `1`, `2`다. `-1`은 맨 뒤, `-2`는 뒤에서 두 번째다.

```text
값       80   95   72
양수      0    1    2
음수     -3   -2   -1
```

리스트는 수정할 수 있다.

```python
scores = [80, 95, 72]

scores[2] = 75
scores.append(88)
removed = scores.pop(0)

print("삭제된 값:", removed)
print("현재 목록:", scores)
```

예상 출력:

```text
삭제된 값: 80
현재 목록: [95, 75, 88]
```

자주 쓰는 메서드:

| 코드 | 하는 일 | 원본 변경 |
|---|---|---|
| `x.append(v)` | 끝에 값 하나 추가 | 예 |
| `x.extend([a, b])` | 여러 값을 이어 붙임 | 예 |
| `x.insert(i, v)` | i 위치 앞에 삽입 | 예 |
| `x.remove(v)` | 첫 번째 값 v 삭제 | 예 |
| `x.pop(i)` | i 위치를 삭제하며 반환 | 예 |
| `x.sort()` | 원본을 정렬 | 예 |
| `sorted(x)` | 정렬된 새 리스트 반환 | 아니요 |

## 3. slice: 구간을 잘라 보기

slice는 `시작:끝:간격` 형식이며 **시작은 포함하고 끝은 제외**한다.

```python
values = [0, 10, 20, 30, 40, 50]

print(values[1:4])
print(values[:3])
print(values[3:])
print(values[::2])
print(values[::-1])
```

예상 출력:

```text
[10, 20, 30]
[0, 10, 20]
[30, 40, 50]
[0, 20, 40]
[50, 40, 30, 20, 10, 0]
```

`values[1:4]`의 길이는 보통 `4 - 1 = 3`이다. 끝을 제외하는 규칙은 train/validation 데이터를 나눌 때도 중요하다.

## 4. tuple: 순서가 있지만 바꿀 수 없는 값 묶음

튜플은 보통 소괄호 `()`로 만든다. 리스트처럼 index와 slice를 쓰지만 원소를 바꿀 수 없다.

```python
image_shape = (3, 224, 224)

channels, height, width = image_shape
print("채널:", channels)
print("높이:", height)
print("너비:", width)
print("첫 두 축:", image_shape[:2])
```

예상 출력:

```text
채널: 3
높이: 224
너비: 224
첫 두 축: (3, 224)
```

배열과 텐서의 `shape`가 튜플로 표현되는 경우가 많다. 값 하나짜리 튜플은 쉼표가 필요하다: `(5,)`. `(5)`는 단순히 괄호로 감싼 정수다.

## 5. dict: 이름표(key)로 값을 찾는 자료구조

딕셔너리는 `{key: value}`로 만든다. 열 이름, 설정값, 한 샘플의 특성을 표현하기 좋다.

```python
sample = {
    "id": "A01",
    "speed": 52.3,
    "is_fault": False,
}

print(sample["speed"])
print(sample.get("temperature", "값 없음"))

sample["speed"] = 55.0
sample["temperature"] = 24.1

print(list(sample.keys()))
print(list(sample.values()))
```

예상 출력:

```text
52.3
값 없음
['id', 'speed', 'is_fault', 'temperature']
['A01', 55.0, False, 24.1]
```

`sample["missing"]`은 키가 없으면 `KeyError`가 난다. 키가 없을 때 기본값이 필요하면 `sample.get("missing", 기본값)`을 쓴다. 다만 반드시 있어야 하는 키를 조용히 기본값으로 대체하면 데이터 오류를 숨길 수 있으므로 목적을 구분한다.

딕셔너리를 반복하면 기본적으로 key를 순회한다. key와 value를 함께 얻으려면 `.items()`를 쓴다.

```python
metrics = {"accuracy": 0.91, "f1": 0.88}

for name, value in metrics.items():
    print(f"{name}: {value:.2f}")
```

예상 출력:

```text
accuracy: 0.91
f1: 0.88
```

## 6. set: 중복 없는 값의 집합

집합은 중복을 제거하거나 포함 여부, 교집합·합집합을 구할 때 편리하다. 순서나 index를 기대하지 않는다.

```python
train_labels = {"normal", "warning", "fault"}
test_labels = {"normal", "fault", "unknown"}

print("공통:", sorted(train_labels & test_labels))
print("전체:", sorted(train_labels | test_labels))
print("test에만:", sorted(test_labels - train_labels))
print("fault 존재:", "fault" in train_labels)
```

예상 출력:

```text
공통: ['fault', 'normal']
전체: ['fault', 'normal', 'unknown', 'warning']
test에만: ['unknown']
fault 존재: True
```

출력 순서를 일정하게 보이려고 `sorted`를 사용했다. 빈 집합은 `{}`가 아니라 `set()`이다. `{}`는 빈 딕셔너리다.

## 7. 중첩 자료구조

실제 데이터는 "샘플 딕셔너리 여러 개를 담은 리스트"처럼 중첩되어 있다.

```python
records = [
    {"id": "A01", "values": [10, 12, 14]},
    {"id": "A02", "values": [7, 9, 8]},
]

print(records[0])
print(records[1]["id"])
print(records[0]["values"][2])
```

예상 출력:

```text
{'id': 'A01', 'values': [10, 12, 14]}
A02
14
```

`records[0]["values"][2]`를 왼쪽부터 읽으면 된다: 리스트의 첫 샘플 → `values` 키의 리스트 → 그 리스트의 세 번째 값.

## 8. 대입과 복사

리스트를 다른 변수에 대입하면 값 전체가 복사되는 것이 아니라 **같은 리스트 객체를 함께 가리킨다**.

```python
original = [1, 2, 3]
same_object = original
copied = original.copy()

same_object[0] = 999
copied[1] = -1

print("원본:", original)
print("같은 객체:", same_object)
print("복사본:", copied)
print("같은 객체인가:", original is same_object)
```

예상 출력:

```text
원본: [999, 2, 3]
같은 객체: [999, 2, 3]
복사본: [1, -1, 3]
같은 객체인가: True
```

`.copy()`는 얕은 복사다. 리스트 안에 다시 리스트가 있으면 내부 리스트는 공유될 수 있다. 완전히 독립된 중첩 구조가 필요할 때 표준 라이브러리 `copy.deepcopy`를 쓴다.

```python
from copy import deepcopy

original = [[1, 2], [3, 4]]
deep = deepcopy(original)
deep[0][0] = 999

print(original)
print(deep)
```

예상 출력:

```text
[[1, 2], [3, 4]]
[[999, 2], [3, 4]]
```

## 9. 흔한 오류와 해결법

| 실수 | 발생하는 문제 | 해결 |
|---|---|---|
| 세 번째 값을 `x[3]`으로 접근 | 실제로 네 번째 값이거나 `IndexError` | 첫 값이 `x[0]`임을 기억 |
| `x[1:3]`에 index 3도 포함된다고 생각 | 예상보다 하나 적음 | slice 끝은 제외 |
| `x = x.sort()` | `x`가 `None`이 됨 | `x.sort()` 또는 `x = sorted(x)` |
| 없는 dict 키를 `d[key]`로 접근 | `KeyError` | 키 검증 또는 의도에 맞게 `.get` |
| set의 첫 원소를 `s[0]`으로 접근 | `TypeError` | set은 순서·index 없음 |
| `copy = original` 후 복사본만 바꾼다고 생각 | 원본도 변경 | list `.copy()` 또는 `deepcopy` |

## 10. 실습문제

1. 리스트 `[5, 10, 15, 20, 25]`에서 첫 값, 마지막 값, 가운데 세 값 `[10, 15, 20]`을 각각 출력하라.
2. 리스트 `[3, 1, 2]`의 원본은 유지하면서 정렬된 새 리스트를 만들어 둘 다 출력하라.
3. 튜플 `(32, 10, 5)`를 `batch`, `time`, `features` 세 변수로 풀어 출력하라.
4. `{"lr": 0.01, "epochs": 20}`에 `"batch_size": 32`를 추가하고 `lr`을 `0.001`로 바꾸라.
5. 리스트 `['A', 'B', 'A', 'C', 'B']`에서 중복을 제거하고 알파벳 순으로 출력하라.
6. `[{"name": "x", "scores": [80, 90]}, {"name": "y", "scores": [70, 75]}]`에서 `y`의 두 번째 점수를 출력하라.
7. 중첩 리스트 `[[1, 2], [3, 4]]`를 완전 복사한 뒤 복사본의 `4`를 `40`으로 바꾸고 원본이 유지되는지 확인하라.

<details>
<summary>입문 3강 정답·해설 펼치기</summary>

### 1번

```python
values = [5, 10, 15, 20, 25]
print(values[0])
print(values[-1])
print(values[1:4])
```

예상 출력은 차례로 `5`, `25`, `[10, 15, 20]`이다. 슬라이싱에서 끝 위치인 인덱스 4는 포함되지 않는다.

### 2번

```python
original = [3, 1, 2]
ordered = sorted(original)
print(original)
print(ordered)
```

예상 출력: `[3, 1, 2]`와 `[1, 2, 3]`. `sorted`는 새 리스트를 반환한다.

### 3번

```python
shape = (32, 10, 5)
batch, time, features = shape
print(batch, time, features)
```

예상 출력: `32 10 5`. 이처럼 튜플의 값을 여러 변수에 나누어 대입하는 것을 튜플 언패킹(tuple unpacking)이라고 한다.

### 4번

```python
config = {"lr": 0.01, "epochs": 20}
config["batch_size"] = 32
config["lr"] = 0.001
print(config)
```

예상 출력: `{'lr': 0.001, 'epochs': 20, 'batch_size': 32}`.

### 5번

```python
labels = ['A', 'B', 'A', 'C', 'B']
unique_labels = sorted(set(labels))
print(unique_labels)
```

예상 출력: `['A', 'B', 'C']`. set으로 중복을 제거하고 sorted로 표시 순서를 정했다.

### 6번

```python
records = [
    {"name": "x", "scores": [80, 90]},
    {"name": "y", "scores": [70, 75]},
]
print(records[1]["scores"][1])
```

예상 출력: `75`.

### 7번

```python
from copy import deepcopy

original = [[1, 2], [3, 4]]
copied = deepcopy(original)
copied[1][1] = 40
print(original)
print(copied)
```

예상 출력: 원본은 `[[1, 2], [3, 4]]`, 복사본은 `[[1, 2], [3, 40]]`.

</details>

## 완료 기준

- [ ] 네 자료구조를 각각 언제 쓰는지 한 문장으로 설명한다.
- [ ] `x[1:4]`, `x[-1]`, `x[::2]`의 결과를 실행 전 예상할 수 있다.
- [ ] 중첩된 list와 dict에서 값을 꺼낼 수 있다.
- [ ] `b = a`와 `b = a.copy()`가 다르게 동작하는 이유를 안다.
- [ ] 실습 7문제 중 6문제 이상을 정답 없이 풀었다.

---

# 입문 4강. 조건문·반복문·함수와 예외 처리

## 1. 왜 필요한가

데이터 값에 따라 다른 처리를 하고, 여러 행에 같은 계산을 반복하고, 자주 쓰는 로직을 함수로 묶어야 실제 분석 코드를 만들 수 있다. 시험의 Process 문항도 대부분 "입력을 받아 요구된 결과를 반환하는 함수" 형태로 생각하면 풀기 쉬워진다.

이 강의를 마치면 다음을 할 수 있다.

- `if`/`elif`/`else`로 조건에 따라 분기한다.
- `for`, `range`, `enumerate`, `zip`으로 반복한다.
- 입력과 반환값이 분명한 함수를 작성한다.
- `try`/`except`를 꼭 필요한 오류에만 사용한다.
- `assert`로 코드의 결과가 필요한 조건을 만족하는지 확인한다.

## 2. 들여쓰기와 코드 블록

Python은 중괄호 대신 **들여쓰기**로 코드 묶음을 나타낸다. 보통 공백 4칸을 사용한다. 조건문 끝의 콜론 `:`도 빠뜨리지 않는다.

```python
score = 84

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "D"

print(grade)
```

예상 출력:

```text
B
```

위에서부터 조건을 확인하고 처음 참인 블록 하나만 실행한다. 그러므로 범위가 겹친다면 더 엄격한 조건을 먼저 둔다.

```python
value = 12

if (value >= 0) and (value <= 10):
    print("0~10 범위")
else:
    print("범위 밖")
```

예상 출력: `범위 밖`.

## 3. for 반복문

`for 변수 in 반복가능한값:`은 값을 하나씩 꺼내 블록을 반복한다.

```python
scores = [80, 95, 72]
total = 0

for score in scores:
    total += score
    print("현재 점수:", score, "누적:", total)

mean = total / len(scores)
print("평균:", mean)
```

예상 출력:

```text
현재 점수: 80 누적: 80
현재 점수: 95 누적: 175
현재 점수: 72 누적: 247
평균: 82.33333333333333
```

인덱스와 값을 함께 사용하려면 `enumerate`를 쓴다.

```python
names = ["A", "B", "C"]

for index, name in enumerate(names, start=1):
    print(index, name)
```

예상 출력:

```text
1 A
2 B
3 C
```

두 목록을 같은 위치끼리 묶으려면 `zip`을 쓴다.

```python
names = ["A", "B", "C"]
scores = [81, 92, 77]

for name, score in zip(names, scores):
    print(f"{name}: {score}")
```

예상 출력:

```text
A: 81
B: 92
C: 77
```

`zip`은 더 짧은 쪽이 끝나면 멈춘다. 두 목록 길이가 반드시 같아야 한다면 반복 전 `assert len(names) == len(scores)`로 확인한다.

## 4. range와 while

`range(start, stop, step)`도 stop을 포함하지 않는다.

```python
print(list(range(5)))
print(list(range(2, 8, 2)))

for epoch in range(1, 4):
    print(f"epoch {epoch}")
```

예상 출력:

```text
[0, 1, 2, 3, 4]
[2, 4, 6]
epoch 1
epoch 2
epoch 3
```

`while`은 조건이 참인 동안 반복한다. 조건을 언젠가 거짓으로 만들지 않으면 무한 반복이 된다.

```python
count = 3

while count > 0:
    print(count)
    count -= 1

print("끝")
```

예상 출력:

```text
3
2
1
끝
```

## 5. break, continue, 리스트 컴프리헨션

`break`는 반복 전체를 끝내고, `continue`는 현재 차례만 건너뛴다.

```python
values = [3, -1, 5, -2, 8]
positives = []

for value in values:
    if value < 0:
        continue
    positives.append(value)

print(positives)
```

예상 출력: `[3, 5, 8]`.

간단한 변환·필터는 리스트 컴프리헨션으로 쓸 수 있다. 처음에는 일반 반복문을 이해한 다음 사용한다.

```python
values = [1, 2, 3, 4, 5]
squares_of_even = [value ** 2 for value in values if value % 2 == 0]
print(squares_of_even)
```

예상 출력: `[4, 16]`.

복잡한 조건을 한 줄로 억지로 줄이면 읽기 어려우므로 일반 반복문이 더 낫다.

## 6. 함수: 입력에 따라 결과를 돌려주는 코드 묶음

함수는 `def`로 정의한다. 괄호 안은 입력 **매개변수(parameter)**, `return`은 호출한 곳에 돌려주는 결과다.

```python
def calculate_mean(values):
    """숫자 목록의 산술평균을 반환한다."""
    if len(values) == 0:
        raise ValueError("values는 비어 있을 수 없습니다")
    return sum(values) / len(values)


scores = [80, 90, 70]
result = calculate_mean(scores)
print(result)
```

예상 출력: `80.0`.

함수 이름은 하는 일을 알 수 있게 짓고, 어떤 값을 받아 무엇을 반환하는지 분명히 정한다. `print`와 `return`은 다르다. `print`는 화면에 보여 주지만 호출 결과는 기본적으로 `None`이다. `return`은 값을 바깥으로 보낸다.

```python
def add_and_print(a, b):
    print(a + b)


def add_and_return(a, b):
    return a + b


printed_result = add_and_print(2, 3)
returned_result = add_and_return(2, 3)

print("print 함수의 반환:", printed_result)
print("return 함수의 반환:", returned_result)
```

예상 출력:

```text
5
print 함수의 반환: None
return 함수의 반환: 5
```

기본값 매개변수도 사용할 수 있다.

```python
def normalize_score(score, maximum=100):
    if maximum <= 0:
        raise ValueError("maximum은 양수여야 합니다")
    return score / maximum


print(normalize_score(80))
print(normalize_score(score=15, maximum=20))
```

예상 출력:

```text
0.8
0.75
```

## 7. 예외 처리

예외는 실행 중 발견된 문제다. 모든 오류를 `except:`로 숨기면 잘못된 결과를 알아차리지 못한 채 다음 계산에 사용할 수 있다. **예상하고 처리할 수 있는 구체적인 예외만** 잡는다.

```python
raw_values = ["10", "bad", "30"]
converted = []

for raw in raw_values:
    try:
        number = int(raw)
    except ValueError:
        print(f"숫자로 바꿀 수 없어 제외: {raw}")
        continue
    converted.append(number)

print(converted)
```

예상 출력:

```text
숫자로 바꿀 수 없어 제외: bad
[10, 30]
```

잘못된 입력을 받았다면 엉뚱한 값을 반환하지 말고, 원인을 알 수 있는 예외를 발생시키는 편이 좋다.

```python
def safe_divide(numerator, denominator):
    if denominator == 0:
        raise ValueError("denominator는 0일 수 없습니다")
    return numerator / denominator


try:
    print(safe_divide(10, 0))
except ValueError as error:
    print("입력 오류:", error)
```

예상 출력: `입력 오류: denominator는 0일 수 없습니다`.

## 8. assert로 필요한 조건 확인하기

`assert 조건, 메시지`는 조건이 거짓이면 즉시 `AssertionError`를 낸다. 개발·연습 중 shape, 길이, 값 범위를 확인하기 좋다. 사용자의 잘못된 입력을 친절하게 처리하는 일반 로직을 전부 `assert`로 대체하는 것은 피한다.

```python
predictions = [0, 1, 1]
targets = [0, 1, 0]

assert len(predictions) == len(targets), "예측과 정답 길이가 달라요"
assert all(value in (0, 1) for value in predictions), "예측은 0 또는 1이어야 해요"

print("검사 통과")
```

예상 출력: `검사 통과`.

## 9. 흔한 오류와 해결법

| 오류·실수 | 원인 | 해결 |
|---|---|---|
| `IndentationError` | 블록 들여쓰기 불일치 | 같은 블록은 공백 4칸 통일 |
| `if score = 80:` | 조건에서 대입 기호 사용 | 비교는 `==` |
| 함수 결과가 `None` | `return` 없이 `print`만 함 | 필요한 값을 `return` |
| 반복문이 끝나지 않음 | while 조건을 바꾸지 않음 | 종료 조건과 갱신문 확인 |
| `zip`이 일부만 반복 | 두 입력 길이가 다름 | 길이 assert 또는 의도 확인 |
| `except:`로 모든 오류 무시 | 버그까지 숨김 | 예상 가능한 구체 예외만 처리 |
| 함수가 입력 리스트를 몰래 변경 | 같은 객체를 직접 수정 | 명세상 필요 없으면 복사 후 처리 |

## 10. 실습문제

1. 정수 `-3`이 양수, 0, 음수 중 무엇인지 조건문으로 출력하라.
2. `[3, 7, 2, 9, 4]`에서 5보다 큰 값만 새 리스트에 담아 출력하라.
3. `range`를 사용해 1부터 10까지 짝수의 합을 구하라.
4. 두 리스트 `names = ["A", "B"]`, `scores = [90, 75]`를 `zip`으로 묶어 `A=90`, `B=75` 형식으로 출력하라. 먼저 길이가 같은지 assert하라.
5. 숫자 리스트를 받아 최솟값과 최댓값을 튜플로 반환하는 `min_max` 함수를 작성하라. 빈 리스트에는 `ValueError`를 발생시켜라.
6. 문자열 목록 `["1.5", "x", "2.5"]`을 float로 변환하되 바꿀 수 없는 값은 제외하라.
7. `[1, 2, 3, 4, 5]`에서 홀수의 제곱만 리스트 컴프리헨션으로 만들라.

<details>
<summary>입문 4강 정답·해설 펼치기</summary>

### 1번

```python
number = -3

if number > 0:
    label = "양수"
elif number == 0:
    label = "0"
else:
    label = "음수"

print(label)
```

예상 출력: `음수`.

### 2번

```python
values = [3, 7, 2, 9, 4]
selected = []

for value in values:
    if value > 5:
        selected.append(value)

print(selected)
```

예상 출력: `[7, 9]`.

### 3번

```python
total = 0
for number in range(2, 11, 2):
    total += number
print(total)
```

예상 출력: `30`. stop 11은 포함되지 않아 10까지 생성된다.

### 4번

```python
names = ["A", "B"]
scores = [90, 75]

assert len(names) == len(scores), "길이가 다릅니다"
for name, score in zip(names, scores):
    print(f"{name}={score}")
```

예상 출력은 `A=90`, `B=75` 두 줄이다.

### 5번

```python
def min_max(values):
    if len(values) == 0:
        raise ValueError("빈 리스트는 처리할 수 없습니다")
    return min(values), max(values)


print(min_max([8, 2, 10, 4]))
```

예상 출력: `(2, 10)`. 여러 값을 쉼표로 반환하면 튜플이 된다.

### 6번

```python
raw_values = ["1.5", "x", "2.5"]
numbers = []

for raw in raw_values:
    try:
        numbers.append(float(raw))
    except ValueError:
        continue

print(numbers)
```

예상 출력: `[1.5, 2.5]`.

### 7번

```python
values = [1, 2, 3, 4, 5]
result = [value ** 2 for value in values if value % 2 == 1]
print(result)
```

예상 출력: `[1, 9, 25]`.

</details>

## 완료 기준

- [ ] 들여쓰기와 콜론을 사용해 조건문·반복문을 쓸 수 있다.
- [ ] `range`의 끝이 포함되지 않는다는 것을 안다.
- [ ] `print`와 `return`의 차이를 코드로 보여 줄 수 있다.
- [ ] 빈 입력과 잘못된 형 변환을 함수에서 안전하게 처리할 수 있다.
- [ ] 실습 7문제 중 6문제 이상을 정답 없이 풀었다.
# 입문 5강. NumPy 배열 만들기: 모양과 자료형 확인

## 1. 왜 필요한가

Python 리스트만으로도 숫자를 담을 수 있지만, 머신러닝은 수천~수백만 개 숫자를 같은 방식으로 계산한다. NumPy는 같은 자료형의 숫자를 다차원 배열에 저장하고 빠르게 **벡터화 연산**을 수행한다. pandas와 PyTorch에서도 배열의 모양(shape)과 자료형(dtype)을 확인한다. NumPy에서 이 두 개념을 익혀 두면 다른 라이브러리를 배우기도 수월하다.

이 강의를 마치면 다음을 할 수 있다.

- `import numpy as np`가 무엇인지 설명한다.
- `np.array`, `zeros`, `ones`, `arange`, `linspace`로 배열을 만든다.
- `ndim`, `shape`, `size`, `dtype`을 읽는다.
- 배열끼리 원소별 연산하고 집계값을 구한다.
- Python list 연산과 NumPy 배열 연산의 차이를 설명한다.

## 2. ndarray는 같은 자료형의 N차원 배열

```python
import numpy as np

values = np.array([10, 20, 30], dtype=np.int64)

print(values)
print(type(values))
print("차원 수:", values.ndim)
print("모양:", values.shape)
print("원소 수:", values.size)
print("자료형:", values.dtype)
```

예상 출력:

```text
[10 20 30]
<class 'numpy.ndarray'>
차원 수: 1
모양: (3,)
원소 수: 3
자료형: int64
```

`(3,)`은 길이 3인 1차원 배열이다. `(3, 1)`은 3행 1열인 2차원 배열이므로 서로 다르다.

```python
import numpy as np

vector = np.array([10, 20, 30])
column = np.array([[10], [20], [30]])

print("vector:", vector.shape, vector.ndim)
print("column:", column.shape, column.ndim)
```

예상 출력:

```text
vector: (3,) 1
column: (3, 1) 2
```

### shape를 문장으로 읽는 습관

`matrix.shape == (2, 3)`이라면 "2개 행, 각 행에 3개 열"이라고 읽는다.

```python
import numpy as np

matrix = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ],
    dtype=np.float32,
)

print(matrix)
print(matrix.shape)
print(matrix.dtype)
```

예상 출력:

```text
[[1. 2. 3.]
 [4. 5. 6.]]
(2, 3)
float32
```

## 3. 배열을 만드는 여러 방법

```python
import numpy as np

a = np.zeros((2, 3), dtype=np.float32)
b = np.ones((2, 3), dtype=np.float32)
c = np.full((2, 3), 7, dtype=np.int64)
d = np.arange(0, 10, 2)
e = np.linspace(0.0, 1.0, num=5)

print("zeros:\n", a)
print("ones:\n", b)
print("full:\n", c)
print("arange:", d)
print("linspace:", e)
```

예상 출력:

```text
zeros:
 [[0. 0. 0.]
 [0. 0. 0.]]
ones:
 [[1. 1. 1.]
 [1. 1. 1.]]
full:
 [[7 7 7]
 [7 7 7]]
arange: [0 2 4 6 8]
linspace: [0.   0.25 0.5  0.75 1.  ]
```

- `np.arange(start, stop, step)`: Python `range`처럼 stop을 제외한다.
- `np.linspace(start, stop, num)`: 시작과 끝을 기본적으로 모두 포함해 같은 간격의 `num`개 값을 만든다.
- shape를 받는 함수에는 `(행, 열)` 튜플을 전달한다.

예제를 다시 실행해도 같은 난수가 나오도록 난수 생성기의 초기값인 시드(seed)를 고정한다. 시드와 NumPy 실행 환경이 같으면 같은 순서의 난수를 얻을 수 있다.

```python
import numpy as np

rng = np.random.default_rng(42)
random_values = rng.integers(low=0, high=10, size=(2, 4))

print(random_values)
print(random_values.shape)
```

예상 출력:

```text
[[0 7 6 4]
 [4 8 0 6]]
(2, 4)
```

`high=10`은 포함되지 않아 0 이상 10 미만의 정수다. 재현성은 학습·검증 결과를 비교할 때 중요하다.

## 4. dtype: 숫자를 저장하는 방식

NumPy 배열은 보통 모든 원소가 같은 dtype이다.

```python
import numpy as np

integers = np.array([1, 2, 3], dtype=np.int64)
features = integers.astype(np.float32)
flags = np.array([True, False, True], dtype=np.bool_)

print(integers, integers.dtype)
print(features, features.dtype)
print(flags, flags.dtype)
```

예상 출력:

```text
[1 2 3] int64
[1. 2. 3.] float32
[ True False  True] bool
```

머신러닝에서 자주 보는 dtype:

| dtype | 대표 용도 | 주의 |
|---|---|---|
| `int64` | 정수로 나타낸 클래스 번호와 인덱스 | 소수 저장 불가 |
| `float32` | 신경망 입력·가중치 | PyTorch 기본 모델과 잘 맞음 |
| `float64` | pandas·NumPy의 일반 실수 계산 | torch float32 모델과 섞으면 dtype 불일치 가능 |
| `bool` | 조건에 맞는 위치를 표시하는 마스크 | `True`/`False` |

문자열과 숫자를 섞어 배열을 만들면 전체가 문자열 dtype으로 바뀔 수 있다.

```python
import numpy as np

mixed = np.array([1, 2, "3"])
print(mixed)
print(mixed.dtype)

numeric = mixed.astype(np.int64)
print(numeric + 1)
```

예상 출력의 dtype 표시는 문자열 길이에 따라 `<U...`처럼 보일 수 있다.

```text
['1' '2' '3']
<U...
[2 3 4]
```

## 5. 리스트 연산과 배열 연산의 차이

```python
import numpy as np

python_list = [1, 2, 3]
numpy_array = np.array([1, 2, 3])

print("list * 2:", python_list * 2)
print("array * 2:", numpy_array * 2)
print("array + 10:", numpy_array + 10)
```

예상 출력:

```text
list * 2: [1, 2, 3, 1, 2, 3]
array * 2: [2 4 6]
array + 10: [11 12 13]
```

리스트 `* 2`는 목록을 반복하지만 배열 `* 2`는 모든 원소를 2배 한다. 배열끼리도 기본적으로 **원소별(element-wise)** 계산이다.

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([10.0, 20.0, 30.0])

print(a + b)
print(a * b)
print(a / b)
print(a ** 2)
```

예상 출력:

```text
[11. 22. 33.]
[10. 40. 90.]
[0.1 0.1 0.1]
[1. 4. 9.]
```

행렬곱은 원소별 곱 `*`와 다르며 `@`를 쓴다.

```python
import numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0]])
w = np.array([[2.0], [1.0]])

print("원소별 제곱:\n", x * x)
print("행렬곱:\n", x @ w)
print("행렬곱 shape:", (x @ w).shape)
```

예상 출력:

```text
원소별 제곱:
 [[ 1.  4.]
 [ 9. 16.]]
행렬곱:
 [[ 4.]
 [10.]]
행렬곱 shape: (2, 1)
```

## 6. 집계 함수

```python
import numpy as np

values = np.array([2.0, 4.0, 6.0, 8.0])

print("합:", values.sum())
print("평균:", values.mean())
print("최솟값:", values.min())
print("최댓값:", values.max())
print("표준편차:", values.std())
print("최댓값 위치:", values.argmax())
```

예상 출력:

```text
합: 20.0
평균: 5.0
최솟값: 2.0
최댓값: 8.0
표준편차: 2.23606797749979
최댓값 위치: 3
```

마지막 index가 3인 이유는 0부터 세기 때문이다. 다차원 집계의 `axis`는 다음 강에서 자세히 다룬다.

## 7. 조건을 배열 전체에 적용

```python
import numpy as np

scores = np.array([55, 80, 91, 67, 88])
mask = scores >= 80

print(mask)
print(scores[mask])
print("80점 이상 수:", mask.sum())
```

예상 출력:

```text
[False  True  True False  True]
[80 91 88]
80점 이상 수: 3
```

불리언 배열의 합을 구할 때 `True`는 1처럼 취급된다. 이 배열로 조건에 맞는 원소만 고르는 방법을 <strong>불리언 인덱싱(Boolean indexing)</strong>이라고 한다.

여러 조건은 각각 괄호로 감싸고 `&`(그리고), `|`(또는), `~`(아님)을 쓴다.

```python
import numpy as np

scores = np.array([55, 80, 91, 67, 88])
selected = scores[(scores >= 70) & (scores < 90)]
print(selected)
```

예상 출력: `[80 88]`.

## 8. 흔한 오류와 해결법

| 오류·실수 | 원인 | 해결 |
|---|---|---|
| `np`가 정의되지 않음 | import 셀 미실행 | 코드 앞에 `import numpy as np` |
| `(3,)`과 `(3,1)`을 같게 생각 | 차원 수가 다름 | `shape`, `ndim`을 모두 출력 |
| 배열 `*`를 행렬곱으로 생각 | `*`는 원소별 | 행렬곱은 `@` |
| 정수 배열에 `0.5` 대입 후 0이 됨 | int dtype은 소수 저장 불가 | `astype(np.float32)` |
| 서로 다른 길이 배열 덧셈 실패 | shape가 호환되지 않음 | 두 shape를 출력하고 의도 확인 |
| `(x > 0) and (x < 5)` | 배열에 Python `and` 사용 | `(x > 0) & (x < 5)` |
| `np.arange(1, 5)`에 5 포함 기대 | stop 제외 | 필요하면 stop을 6으로 |

## 9. 실습문제

1. 0부터 9까지의 정수 배열을 만들고 shape, ndim, size, dtype을 출력하라.
2. 모양이 `(3, 2)`이고 자료형이 float32이며 모든 원소가 1인 배열을 만들어라.
3. 0부터 1까지 양 끝을 포함한 값 6개를 균일하게 만들라.
4. 배열 `[1, 2, 3, 4]`을 float32로 변환하고 각 원소를 10으로 나눈 결과를 출력하라.
5. 배열 `[10, 20, 30]`과 `[1, 2, 3]`의 원소별 곱과 내적(`@`)을 각각 구하라.
6. 점수 `[45, 70, 85, 90, 60]`에서 70 이상 90 이하인 값만 선택하라.
7. 시드를 7로 설정한 난수 생성기로 0 이상 100 미만인 정수 12개를 만들어라. 이를 `(3, 4)` 모양의 배열로 만들고 모양과 평균을 출력하라.

<details>
<summary>입문 5강 정답·해설 펼치기</summary>

### 1번

```python
import numpy as np

x = np.arange(10)
print(x)
print(x.shape, x.ndim, x.size, x.dtype)
```

예상 배열은 `[0 1 2 3 4 5 6 7 8 9]`, shape는 `(10,)`, ndim은 `1`, size는 `10`이다. 기본 정수 dtype의 비트 수는 환경에 따라 다를 수 있다.

### 2번

```python
import numpy as np

x = np.ones((3, 2), dtype=np.float32)
print(x)
print(x.shape, x.dtype)
```

예상 shape는 `(3, 2)`, dtype은 `float32`다.

### 3번

```python
import numpy as np

x = np.linspace(0.0, 1.0, num=6)
print(x)
```

예상 출력: `[0.  0.2 0.4 0.6 0.8 1. ]`.

### 4번

```python
import numpy as np

x = np.array([1, 2, 3, 4]).astype(np.float32)
result = x / 10
print(result)
print(result.dtype)
```

예상 출력은 `[0.1 0.2 0.3 0.4]`, dtype은 `float32`다.

### 5번

```python
import numpy as np

a = np.array([10, 20, 30])
b = np.array([1, 2, 3])
print("원소별:", a * b)
print("내적:", a @ b)
```

예상 출력: 원소별은 `[10 40 90]`, 내적은 `140`이다.

### 6번

```python
import numpy as np

scores = np.array([45, 70, 85, 90, 60])
selected = scores[(scores >= 70) & (scores <= 90)]
print(selected)
```

예상 출력: `[70 85 90]`.

### 7번

```python
import numpy as np

rng = np.random.default_rng(7)
x = rng.integers(0, 100, size=(3, 4))
print(x)
print("shape:", x.shape)
print("평균:", x.mean())
```

배열의 모양이 `(3, 4)`인지 확인한다. 같은 NumPy 난수 생성기와 실행 환경에서 시드 7을 사용하면 배열의 값도 다시 얻을 수 있다.

</details>

## 완료 기준

- [ ] 1차원 `(F,)`와 2차원 `(B, F)` shape를 문장으로 읽을 수 있다.
- [ ] `ndim`, `shape`, `size`, `dtype`의 차이를 설명한다.
- [ ] list와 ndarray의 `* 2` 결과가 왜 다른지 안다.
- [ ] 원소별 곱과 행렬곱을 구분한다.
- [ ] Boolean mask로 범위 조건의 원소를 선택할 수 있다.
- [ ] 실습 7문제 중 6문제 이상을 정답 없이 풀었다.

---

# 입문 6강. 배열과 텐서의 축·모양 변경과 브로드캐스팅

## 1. 왜 필요한가

머신러닝 오류의 상당수는 값 자체보다 shape와 축을 잘못 이해해 생긴다. "행별 평균"과 "열별 평균"을 뒤집거나, `(B,)`와 `(B, 1)`을 섞거나, 슬라이스가 원본과 메모리를 공유한다는 사실을 놓치면 오류 메시지 없이도 잘못된 결과가 나올 수 있다. 이번 강의는 배열을 안전하게 자르고 모양을 바꾸는 법을 다룬다.

이 강의를 마치면 다음을 할 수 있다.

- 2차원 이상의 배열을 행·열 기준으로 선택한다.
- `axis=0`, `axis=1` 집계 결과의 shape를 예상한다.
- `reshape`, `transpose`의 역할을 구분한다.
- broadcasting 가능 여부를 뒤쪽 축부터 판단한다.
- view와 copy를 구분해 원본 변경을 통제한다.
- `[B,T,F]`를 말로 읽고 PyTorch의 `permute`와 broadcasting을 구분한다.

### 먼저 읽기: shape는 값이 아니라 배열의 크기 설명이다

`[B,F]`를 처음 만났다면 모델 이름부터 외우지 말자. **텐서(tensor)는 숫자를 여러 축으로 배열한 것**이다. NumPy의 `ndarray`와 PyTorch의 `Tensor`는 서로 다른 자료형이지만, 둘 다 shape로 각 축의 크기를 읽는다. 아래 NumPy 기초 뒤에 나오는 PyTorch 예제는 `import torch`로 구분했다. 여기서는 모델을 학습하지 않으므로 GPU가 필요하지 않다.

```python
import numpy as np

# 한 행은 자동차 한 대, 두 열은 온도와 속도다.
x = np.array([[80., 60.], [90., 70.], [85., 50.]])
print(x.shape)   # (3, 2): 자동차 3대, 각 자동차의 특성 2개
print(x.ndim)    # 2: 축이 두 개
print(x.size)    # 6: 숫자는 총 여섯 개
```

`(3,2)`는 값 3과 2가 들어 있다는 뜻이 아니다. **3행 × 2열로 숫자가 들어 있다는 설명**이다. 책의 `[3,2]`, NumPy의 `(3,2)`, PyTorch의 `torch.Size([3,2])`는 이 문맥에서 같은 모양을 가리킨다. 대괄호로 적은 배열 크기는 설명용이지, 텐서를 생성하는 코드가 아니다.

| 기호 | 이름 | 문장으로 읽기 |
|---|---|---|
| B | Batch size | 한 번에 모델에 넣는 샘플 개수 |
| F | Features | 샘플 하나 또는 한 시점의 특성 개수: 온도·압력·진동 등 |
| T | Time steps | 한 기록 안의 시간 단계 수 |
| C | Channels | 입력 채널 수: RGB 이미지라면 3 |
| H, W | Height, Width | 이미지의 세로·가로 픽셀 수 |
| K | Classes | 여기서는 분류할 클래스 수 |

전체 데이터가 1,000개여도 32개씩 처리하면 보통 `B=32`다. 마지막 배치는 더 작을 수 있다. 이 교재에서는 전체 샘플 수를 `N`, 현재 배치 크기를 `B`로 구분한다. 문서에 따라 `N`을 배치 기호로 쓰기도 하므로 **글자 자체보다 그 축의 의미**를 확인한다. 또한 텐서는 축의 이름을 자동으로 알아내지 않는다. `[32,60,3]`만 보고 시간과 특성의 의미를 결정할 수는 없고, 데이터를 만든 과정과 문제 명세를 함께 읽어야 한다.

### 모델별 shape를 한 문장으로 읽기

| 사용하는 곳 | 모양과 예시 | 어떻게 읽는가 |
|---|---|---|
| 표 형태 데이터를 다루는 MLP 입력 | `[B,F]` → `[32,5]` | 자동차 32대 × 자동차마다 특성 5개 |
| 일정 길이로 잘라 묶은 시계열 배치 | `[B,T,F]` → `[32,60,3]` | 주행 기록 32개 × 기록마다 60시점 × 시점마다 특성 3개 |
| Conv1d 입력 | `[B,F,T]` → `[32,3,60]` | 특성 3개를 채널로 두고, 각 채널의 시간 60개를 따라 처리 |
| RNN·LSTM·GRU 입력 | `[B,T,F]` → `[32,60,3]` | **`batch_first=True`일 때** 배치·시간·특성 순서 |
| 이미지 Conv2d 입력 | `[B,C,H,W]` → `[32,3,64,64]` | RGB 이미지 32장 × 채널 3개 × 세로 64 × 가로 64 |
| 샘플당 하나를 분류하는 출력 | `[B,K]` → `[32,3]` | 샘플 32개 × 정상·고장A·고장B에 대한 점수 3개 |

MLP는 기본적인 신경망, Conv1d·Conv2d는 각각 한 축·두 축을 따라 패턴을 찾는 합성곱 층, RNN은 순서를 처리하는 신경망이다. 지금은 내부 수식보다 **어떤 순서로 숫자를 넣어야 하는지**에 집중한다. `Linear` 자체는 마지막 축에 적용되므로 모든 MLP가 반드시 2차원 입력만 받는다는 뜻은 아니다. 위 표는 샘플당 하나를 예측하는 기본 구성이다.

**시계열 CSV 원본과 시계열 배치는 다르다.** 한 차량의 원본 기록은 `[전체 시점 수,F]`인 긴 표일 수 있다. 이를 과거 60개 시점씩 잘라 window(한 샘플)를 만든 뒤, 샘플 32개를 모으면 `[32,60,F]`다. 이미 window로 제공된 데이터에 다시 window를 만들지는 않는다. 여러 차량을 다룰 때는 차량 경계를 넘겨 묶지 않는다.

**RGB 이미지 한 장은 숫자 표 세 장을 포갠 것**으로 생각하자. R·G·B 채널마다 `[H,W]` 표가 있다. PIL 이미지를 NumPy로 바꾸면 흔히 `[H,W,C]`, PyTorch Conv2d에 배치로 넣을 때는 `[B,C,H,W]`를 쓴다. PIL의 `image.size`는 `(W,H)` 순서라는 점도 다르다. 색 순서와 0–255/0–1 범위는 shape만으로 알 수 없다.

**RNN의 조건:** `batch_first=True`가 없으면 기본 입력 순서는 `[T,B,F]`다. 이 옵션은 시퀀스의 입력과 출력에 적용되며, 마지막 은닉 상태 `h_n`과 LSTM의 `c_n`까지 배치 우선으로 바꾸지는 않는다. 그 상태들은 `[층 수 × 방향 수,B,은닉 크기]`다. 여기서 은닉 크기는 이미지 높이와 다른 개념이다.

**logit은 확률이 아니라 모델이 내놓은 원점수**다. 음수도 가능하고 합이 1일 필요가 없다. 샘플마다 점수가 가장 큰 클래스 번호를 고르는 것이 `argmax(dim=1)`이다. `dim=1`은 이 예제에서 클래스 축이고, 번호는 0부터 시작한다.

```python
import torch

# 열 순서: 정상(0), 고장A(1), 고장B(2)
logits = torch.tensor([[2.1, -0.5, 0.8], [-1.0, 3.0, 0.5]])
pred = logits.argmax(dim=1)
probabilities = logits.softmax(dim=1)
print(tuple(logits.shape))       # (2, 3)
print(pred.tolist())             # [0, 1]
print(tuple(pred.shape))         # (2,)
assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
```

`softmax`는 클래스 점수들을 합이 1인 확률로 바꾸는 연산이다. 학습할 때 `CrossEntropyLoss`에는 확률로 변환하기 전의 로짓을 넣고, 제출이 클래스 번호인지 확률인지는 별도로 확인한다. **모델 입력의 크기와 모델 출력의 크기를 구분해서 익힌다.**

읽는 순서: 지금은 shape의 의미를 익히고, **5절에서 축 변경 → 6절에서 broadcasting → 10절에서 직접 풀이**로 확인한다. 모델의 자세한 구조는 본과정에서 배운다.

## 2. 2차원 배열의 인덱싱과 슬라이싱

2차원 배열은 `array[행, 열]` 순서로 접근한다.

```python
import numpy as np

x = np.array(
    [
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ]
)

print("한 원소:", x[1, 2])
print("첫 행:", x[0, :])
print("두 번째 열:", x[:, 1])
print("부분 배열:\n", x[1:, 1:3])
```

예상 출력:

```text
한 원소: 22
첫 행: [10 11 12 13]
두 번째 열: [11 21 31]
부분 배열:
 [[21 22]
 [31 32]]
```

`x[:, 1]`은 크기가 `(3,)`인 1차원 결과다. 열 차원을 유지하고 `(3, 1)`을 원한다면 `x[:, 1:2]`처럼 슬라이싱을 사용한다.

```python
import numpy as np

x = np.arange(12).reshape(3, 4)
one_dim = x[:, 1]
two_dim = x[:, 1:2]

print(one_dim.shape)
print(two_dim.shape)
print(two_dim)
```

예상 출력:

```text
(3,)
(3, 1)
[[1]
 [5]
 [9]]
```

## 3. axis는 "연산을 적용할 축"

`axis`는 항상 축을 없앤다는 뜻이 아니다. `sum`, `mean` 같은 **축약(reduction)** 연산에서는 지정한 축의 값들을 모아 계산하므로 그 축이 사라진다(`keepdims=True`이면 길이 1로 남긴다). 반면 `concatenate`에서는 지정한 축을 따라 배열을 이어 붙여 그 축의 길이가 늘어난다. 어떤 함수에 axis를 주는지 먼저 확인하자.

shape `(3, 4)` 배열을 3행 4열 표라고 생각하자.

```text
axis=0 방향으로 모으면 행 축이 사라짐 → 열마다 하나 → shape (4,)
axis=1 방향으로 모으면 열 축이 사라짐 → 행마다 하나 → shape (3,)
```

```python
import numpy as np

x = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ],
    dtype=np.float32,
)

print("전체 평균:", x.mean())
print("열별 평균 axis=0:", x.mean(axis=0))
print("행별 평균 axis=1:", x.mean(axis=1))
print("입력 shape:", x.shape)
print("axis=0 결과 shape:", x.mean(axis=0).shape)
print("axis=1 결과 shape:", x.mean(axis=1).shape)
```

예상 출력:

```text
전체 평균: 6.5
열별 평균 axis=0: [5. 6. 7. 8.]
행별 평균 axis=1: [ 2.5  6.5 10.5]
입력 shape: (3, 4)
axis=0 결과 shape: (4,)
axis=1 결과 shape: (3,)
```

집계 후 차원을 유지해야 broadcasting하기 쉬운 경우 `keepdims=True`를 쓴다.

```python
import numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0]])
row_mean = x.mean(axis=1, keepdims=True)
centered = x - row_mean

print("행 평균 shape:", row_mean.shape)
print(row_mean)
print("행별 중심화:\n", centered)
```

예상 출력:

```text
행 평균 shape: (2, 1)
[[1.5]
 [3.5]]
행별 중심화:
 [[-0.5  0.5]
 [-0.5  0.5]]
```

## 4. reshape: 원소 수를 유지한 채 모양 바꾸기

`reshape`는 원소의 총 개수를 바꾸지 않는다.

```python
import numpy as np

x = np.arange(12)
matrix = x.reshape(3, 4)
auto_rows = x.reshape(-1, 3)

print(matrix)
print("matrix shape:", matrix.shape)
print(auto_rows)
print("auto_rows shape:", auto_rows.shape)
```

예상 출력:

```text
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
matrix shape: (3, 4)
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
auto_rows shape: (4, 3)
```

`-1`은 "전체 원소 수에 맞게 이 축 크기를 자동 계산"하라는 뜻이며 한 번만 쓸 수 있다. 원소 12개를 `(5, 3)`으로 바꿀 수는 없다.

차원을 추가하거나 제거하는 방법도 자주 쓴다.

```python
import numpy as np

x = np.array([10, 20, 30])
column = x[:, None]
batch = x[None, :]
restored = np.squeeze(column, axis=1)

print("원본:", x.shape)
print("열 벡터:", column.shape)
print("배치 축 추가:", batch.shape)
print("복원:", restored.shape)
```

예상 출력:

```text
원본: (3,)
열 벡터: (3, 1)
배치 축 추가: (1, 3)
복원: (3,)
```

## 5. transpose: 축 순서 바꾸기

`reshape`는 모양을 새로 해석하고, `transpose`는 축의 순서를 바꾼다.

```python
import numpy as np

x = np.arange(6).reshape(2, 3)
transposed = x.T

print("원본:\n", x)
print("전치:\n", transposed)
print(x.shape, "->", transposed.shape)
```

예상 출력:

```text
원본:
 [[0 1 2]
 [3 4 5]]
전치:
 [[0 3]
 [1 4]
 [2 5]]
(2, 3) -> (3, 2)
```

3차원 이상에서는 바꾸고 싶은 축 순서를 명시한다. 일정 길이로 잘라 묶은 시계열 `[B, T, F]`를 Conv1d 입력 형식인 `[B, F, T]`로 바꾸는 예다. Conv1d는 `[배치,채널,길이]`를 받으므로, 여기서는 특성 F를 채널로, 시간 T를 길이로 사용한다.

```python
import numpy as np

x = np.zeros((32, 20, 5), dtype=np.float32)  # [B, T, F]
conv1d_input = x.transpose(0, 2, 1)          # [B, F, T]

print(x.shape)
print(conv1d_input.shape)
```

예상 출력:

```text
(32, 20, 5)
(32, 5, 20)
```

### PyTorch에서는 permute: 숫자 0, 2, 1의 뜻

PyTorch에서 전체 축의 순서를 지정할 때는 `permute`를 쓴다. 축 번호는 0부터 시작한다. **`permute(0,2,1)`은 크기를 0·2·1로 바꾸라는 뜻이 아니라, 기존 축을 0번·2번·1번 순서로 놓으라는 뜻**이다.

| 구분 | 첫 번째 축 | 두 번째 축 | 세 번째 축 |
|---|---|---|---|
| 기존 축 번호 | 0 | 1 | 2 |
| 기존 의미 | B: 배치 | T: 시간 | F: 특성 |
| 가져올 기존 축 번호 | 0 | 2 | 1 |
| 변경 후 의미 | B: 배치 | F: 특성 | T: 시간 |

아래는 B=1, T=3, F=2인 아주 작은 예다. 표는 배치의 첫 번째 샘플만 펼쳐 보인다.

| 변경 전: 시간별 기록 | 특성 A | 특성 B |
|---|---:|---:|
| t1 | 1 | 10 |
| t2 | 2 | 20 |
| t3 | 3 | 30 |

| 변경 후: 특성별 시간 기록 | t1 | t2 | t3 |
|---|---:|---:|---:|
| 특성 A | 1 | 2 | 3 |
| 특성 B | 10 | 20 | 30 |

```python
import torch

x = torch.tensor([[[1., 10.], [2., 20.], [3., 30.]]])
conv_input = x.permute(0, 2, 1)
wrong = x.reshape(1, 2, 3)
print(tuple(x.shape), "->", tuple(conv_input.shape))
print("permute:", conv_input.tolist())
print("reshape:", wrong.tolist())
assert conv_input[0, 0].tolist() == [1., 2., 3.]
assert not torch.equal(conv_input, wrong)
```

예상 출력:

```text
(1, 3, 2) -> (1, 2, 3)
permute: [[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]]
reshape: [[[1.0, 10.0, 2.0], [20.0, 3.0, 30.0]]]
```

두 결과는 shape가 같아도 값의 위치가 다르다. `reshape`는 이 예제에서 나열된 숫자를 세 개씩 다시 묶고, `permute`는 시간·특성의 대응을 유지한 채 축을 바꾼다. **축 교환이 필요한 상황에서 reshape로 크기만 맞추면 안 된다.** `permute`는 원본과 데이터를 공유하는 view이므로 값 복사가 필요하면 별도로 처리한다. NumPy의 `x.transpose(0,2,1)`에 대응하는 PyTorch 표현이 `x.permute(0,2,1)`이며, PyTorch의 `transpose(1,2)`는 두 축만 교환하는 API다. [PyTorch permute 문서](https://docs.pytorch.org/docs/2.7/generated/torch.permute.html)

## 6. 브로드캐스팅: 모양이 다른 배열끼리 계산하기

### 하나의 숫자를 여러 위치에 적용하는 것부터

브로드캐스팅(broadcasting)은 **모양이 다른 배열끼리 원소별 계산을 할 때, 조건이 맞으면 같은 값을 여러 위치에 적용하는 규칙**이다. 축을 교환하는 permute와는 다른 개념이다. NumPy와 PyTorch의 일반적인 원소별 덧셈·뺄셈·곱셈에 같은 규칙이 적용된다. 행렬곱 `@`의 규칙과는 구분한다.

```python
import torch

a = torch.tensor([1, 2, 3])
print((a + 10).tolist())  # [11, 12, 13]
```

숫자 하나인 10을 각각의 원소에 더했다. `[10,10,10]`을 더한 것처럼 이해할 수 있지만, 반복된 입력을 실제로 모두 복사해서 만들 필요는 없다. **입력의 반복 적용이 효율적이라는 뜻이지, 큰 결과 배열도 메모리를 전혀 쓰지 않는다는 뜻은 아니다.**

### 같은 숫자 표에서 permute와 비교하기

앞 절의 첫 샘플 `[3,2]`에서 특성 A에 100, 특성 B에 1000을 더해 보자. 보정값은 `[100,1000]` 한 줄만 주지만 모든 시간에 적용된다.

| 시간 | A에 적용되는 계산 | B에 적용되는 계산 |
|---|---|---|
| t1 | 1 + 100 = 101 | 10 + 1000 = 1010 |
| t2 | 2 + 100 = 102 | 20 + 1000 = 1020 |
| t3 | 3 + 100 = 103 | 30 + 1000 = 1030 |

```python
import torch

x = torch.tensor([[1., 10.], [2., 20.], [3., 30.]])
offset = torch.tensor([100., 1000.])
result = x + offset
print(result.tolist())
print(tuple(x.shape), tuple(offset.shape), tuple(result.shape))
```

예상 출력:

```text
[[101.0, 1010.0], [102.0, 1020.0], [103.0, 1030.0]]
(3, 2) (2,) (3, 2)
```

축 변경에서는 같은 여섯 값의 위치가 바뀌었다. 여기서는 시간·특성의 축 순서는 유지되고, 각 위치의 값에 보정값이 더해졌다.

### 가능 여부는 오른쪽부터 검사한다

NumPy와 PyTorch는 두 배열의 shape를 **오른쪽 축부터** 비교한다. 각 쌍이 같거나 둘 중 하나가 `1`이면 호환된다. 부족한 왼쪽 축에는 크기 1이 있다고 생각한다. 한쪽만 늘어나는 경우뿐 아니라 `[4,1] + [1,3] → [4,3]`처럼 양쪽의 크기 1인 축이 함께 확장되는 경우도 있다. [PyTorch broadcasting 규칙](https://docs.pytorch.org/docs/2.7/notes/broadcasting.html)

```text
데이터     [32, 5]
보정값         [5]
맞춰 읽기  [ 1, 5]

오른쪽: 5와 5는 같다 → 가능
왼쪽: 32와 1은 한쪽이 1이다 → 가능
결과: [32, 5]
```

반면 `[32,5] + [3]`은 마지막 자리의 5와 3이 다르고 둘 다 1이 아니므로 불가능하다. 어떤 행·열에 적용하고 싶은지 결정한 뒤 shape를 맞춘다.

```python
import numpy as np

x = np.array(
    [
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
    ]
)
column_offset = np.array([1.0, 2.0, 3.0])

result = x - column_offset
print(result)
print(x.shape, column_offset.shape, result.shape)
```

예상 출력:

```text
[[ 9. 18. 27.]
 [39. 48. 57.]]
(2, 3) (3,) (2, 3)
```

shape 비교:

```text
x              (2, 3)
column_offset     (3,) → (1, 3)으로 맞춰 읽기
오른쪽 축          3 == 3  → 가능
왼쪽 축           2와 1   → 가능
```

행마다 다른 값을 빼려면 `(2, 1)`이어야 한다.

```python
import numpy as np

x = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
row_offset = np.array([10.0, 40.0])[:, None]

print("offset shape:", row_offset.shape)
print(x - row_offset)
```

예상 출력:

```text
offset shape: (2, 1)
[[ 0. 10. 20.]
 [ 0. 10. 20.]]
```

`(2, 3)`과 `(2,)`는 오른쪽 축 `3`과 `2`가 달라 broadcasting되지 않는다. 이럴 때 값을 억지로 반복하기 전에 어느 축에 적용하려던 값인지 생각하고 `[:, None]`으로 shape를 명확히 한다.

### 실기 함정: [B,1]과 [B]는 같은 모양이 아니다

회귀에서 예측값 네 개와 정답 네 개를 하나씩 짝지어 빼고 싶다고 하자. `[4,1] - [4]`는 오류 없이 `[4,4]`가 된다. 두 번째 입력을 `[1,4]`로 맞춰 보기 때문에 **각 예측이 모든 정답과 비교되는 16개 차이**가 만들어진다.

```python
import torch

pred = torch.tensor([[10.], [20.], [30.], [40.]])  # [4, 1]
target = torch.tensor([11., 19., 31., 39.])       # [4]
wrong_error = pred - target
wrong_mse = wrong_error.square().mean()

target_column = target.unsqueeze(1)  # [4] -> [4, 1]
correct_error = pred - target_column
correct_mse = correct_error.square().mean()

print("잘못된 비교:", tuple(wrong_error.shape), wrong_mse.item())
print("올바른 비교:", tuple(correct_error.shape), correct_mse.item())
print("샘플별 차이:", correct_error.squeeze(1).tolist())
assert pred.shape == target_column.shape
assert correct_mse.item() == 1.0
```

예상 출력:

```text
잘못된 비교: (4, 4) 241.0
올바른 비교: (4, 1) 1.0
샘플별 차이: [-1.0, 1.0, -1.0, 1.0]
```

`unsqueeze(1)`은 1번 위치에 크기 1인 축을 추가한다. 반대로 `pred.squeeze(1)`로 예측을 `[B]`로 맞추는 방법도 있다. 둘 중 **문제에서 요구하는 출력 형식에 맞는 방법**을 고른다. 축 번호 없는 `squeeze()`는 B=1일 때 배치 축까지 지울 수 있어 피한다.

이 예측과 정답을 그대로 `MSELoss`에 넣어도 경고와 함께 의도하지 않은 비교가 일어날 수 있다. 경고를 무시하지 말고 먼저 배열 크기가 의도한 비교에 맞는지 확인한다. `BCEWithLogitsLoss`는 예측과 정답의 배열 크기가 같아야 하며, 정수 클래스 번호를 쓰는 `CrossEntropyLoss`는 로짓 `[B,K]`와 정답 `[B]`처럼 크기가 서로 다른 배열을 받도록 설계되어 있다. **모든 손실함수의 입력을 무조건 같은 모양으로 만들라는 뜻은 아니다.**

### 이미지에서의 응용: 채널별 값은 어느 축에 놓을까?

이미지 배치 `[B,C,H,W]`에 채널별 값 `[C]`를 바로 더하면 마지막 W축과 비교된다. 실패할 수도 있고, W=C인 작은 이미지에서는 엉뚱한 축에 적용되어도 실행될 수 있다. 채널별 값을 `[1,C,1,1]`로 명시한다.

```python
import torch

images = torch.zeros(2, 3, 4, 5)  # B=2, C=3, H=4, W=5
channel_offset = torch.tensor([10., 20., 30.])
result = images + channel_offset.reshape(1, 3, 1, 1)
print(tuple(result.shape))        # (2, 3, 4, 5)
print(result[0, :, 0, 0].tolist()) # [10.0, 20.0, 30.0]
```

여기서 reshape는 길이 3인 채널 값을 유지하며 크기 1인 축을 추가하므로 적절하다. 시계열의 시간·특성 두 축을 교환하려는 앞 절의 상황과 다르다. 이미지 정규화의 `(images-mean)/std`에서도 채널별 mean과 std를 같은 방식으로 맞춘다.

마지막으로 말로 구분해 보자. **shape는 배열의 크기 설명, permute는 축 순서 변경, broadcasting은 계산할 때 값을 반복 적용하는 규칙**이다. 실전에서는 연산 전후 `print(x.shape)`로 예상과 실제를 확인한다.

## 7. view와 copy

NumPy의 기본 slicing은 대개 원본 데이터 메모리를 공유하는 **view**를 만든다. view를 바꾸면 원본도 바뀔 수 있다.

```python
import numpy as np

original = np.array([0, 1, 2, 3, 4])
view = original[1:4]
view[0] = 999

print("원본:", original)
print("view:", view)
print("메모리 공유:", np.shares_memory(original, view))
```

예상 출력:

```text
원본: [  0 999   2   3   4]
view: [999   2   3]
메모리 공유: True
```

원본을 보존하려면 명시적으로 `.copy()`한다.

```python
import numpy as np

original = np.array([0, 1, 2, 3, 4])
independent = original[1:4].copy()
independent[0] = 999

print("원본:", original)
print("복사본:", independent)
print("메모리 공유:", np.shares_memory(original, independent))
```

예상 출력:

```text
원본: [0 1 2 3 4]
복사본: [999   2   3]
메모리 공유: False
```

Boolean·정수 배열을 사용한 advanced indexing은 일반적으로 copy를 만든다. 모든 경우를 외우기보다 원본 보존이 중요하면 `.copy()`를 명시한다.

## 8. 결합과 분할

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
row_joined = np.concatenate([a, b], axis=0)

c = np.array([[10], [20]])
column_joined = np.concatenate([a, c], axis=1)

print("행 추가:\n", row_joined, row_joined.shape)
print("열 추가:\n", column_joined, column_joined.shape)
```

예상 출력:

```text
행 추가:
 [[1 2]
 [3 4]
 [5 6]] (3, 2)
열 추가:
 [[ 1  2 10]
 [ 3  4 20]] (2, 3)
```

`axis=0`으로 결합하면 나머지 축(열 수)이 같아야 하고, `axis=1`이면 행 수가 같아야 한다.

## 9. 흔한 오류와 해결법

| 오류·실수 | 원인 | 해결 |
|---|---|---|
| 행 평균인데 `axis=0` | 없앨 축을 반대로 생각 | 결과 개수·shape를 먼저 예상 |
| `reshape` 후 값 순서도 원하는 축대로 바뀐다고 생각 | reshape는 축 교환이 아님 | 축 순서 변경은 transpose |
| reshape `ValueError` | 원소 수 불일치 | `x.size`와 새 shape의 곱 확인 |
| broadcasting `ValueError` | 오른쪽부터 비교한 축이 다름 | 두 shape를 나란히 적고 1 또는 같음 확인 |
| slice 수정 후 원본까지 변경 | NumPy slice가 view | 원본 보존이면 `.copy()` |
| `squeeze()`가 필요한 축까지 제거 | 크기 1인 모든 축 제거 | `squeeze(axis=...)`로 축 명시 |
| 배열 결합 시 크기 불일치 오류 | 결합하는 축 이외의 크기가 다름 | 각 입력 배열의 크기를 출력해 비교 |

## 10. 실습문제

1. `np.arange(20).reshape(4, 5)`에서 마지막 두 행의 2~4번째 열을 선택하라. 결과 shape도 출력하라.
2. `(3, 4)` 배열의 열별 합과 행별 합을 각각 구하고 결과 shape를 출력하라.
3. 길이 24 배열을 `(2, 3, 4)`로 바꾸고 원소 수가 그대로인지 assert하라.
4. 크기가 `(5,)`인 배열을 `(1, 5)`와 `(5, 1)`로 각각 바꾸라.
5. 크기가 `(2, 3)`인 배열에서 각 열의 평균을 빼 중심화하라. 결과의 열별 평균이 0에 가까운지 확인하라.
6. `(2, 3, 4)`를 `(2, 4, 3)`으로 축 교환하라. reshape가 아닌 transpose를 사용한다.
7. 배열 일부를 복사해 수정하고 원본이 변하지 않았음을 확인하라.

### 추가 실습: shape를 읽고 PyTorch로 확인하기

8. 전체 1,000개 샘플을 32개씩 처리한다. 이번 배치의 `B`는 무엇인가? `[32,60,3]`이 배치·시간·특성 순서의 시계열 데이터라고 주어졌을 때 각 축을 설명하고, RGB 이미지 `[8,3,64,64]`의 각 축도 읽어라. 원본 CSV가 항상 `[B,T,F]`인지 설명하라.
9. `torch.arange(12).reshape(2,3,2)`를 `[B,T,F]` 입력으로 해석하라. Conv1d에 넣을 `[B,F,T]`로 바꾸고, `변경후[1,0,2] == 변경전[1,2,0]`인지 확인하라. 같은 크기로 reshape만 하면 왜 틀리는가?
10. 다음 원소별 덧셈의 결과 배열 크기 또는 실패 이유를 먼저 종이에 쓰고 확인하라: `[4,3]+[3]`, `[4,3]+[4]`, `[4,1]+[3]`, `[2,5,3]+[1,5,1]`. 마지막 식에서 두 번째 배열이 어느 축에 적용되는지도 설명하라.
11. `pred=[[2.],[4.]]`, `target=[3.,5.]`의 원소별 제곱 오차 평균을 계산하라. 그냥 뺀 결과와 shape를 바로잡은 결과를 비교하고, B=1에서도 배치 축이 남는지 확인하라.
12. `[2,3,4,5]` 이미지 배치에서 채널별 평균 `[0.1,0.2,0.3]`을 빼고 싶다. 평균의 shape를 무엇으로 바꿔야 하는가? `images-mean`을 그대로 쓰면 어느 축과 비교되는가?
13. `logits=[[0.,2.,1.],[3.,0.,1.]]`의 클래스 번호와 출력 shape를 구하라. 같은 배치에 `RNN(input_size=3, hidden_size=4, batch_first=True)`로 길이 5의 시계열을 넣을 때 입력·시점별 출력·마지막 은닉 상태의 모양를 각각 예상하라. 층은 1개, 방향은 단방향이다.

<details>
<summary>입문 6강 정답·해설 펼치기</summary>

### 1번

```python
import numpy as np

x = np.arange(20).reshape(4, 5)
selected = x[-2:, 1:4]
print(x)
print(selected)
print(selected.shape)
```

예상 selected는 `[[11, 12, 13], [16, 17, 18]]`, shape는 `(2, 3)`이다.

### 2번

```python
import numpy as np

x = np.arange(12).reshape(3, 4)
column_sum = x.sum(axis=0)
row_sum = x.sum(axis=1)
print(column_sum, column_sum.shape)
print(row_sum, row_sum.shape)
```

예상 shape는 각각 `(4,)`, `(3,)`이다. axis로 지정한 축이 사라진다고 생각한다.

### 3번

```python
import numpy as np

x = np.arange(24)
y = x.reshape(2, 3, 4)
assert x.size == y.size
print(y.shape, y.size)
```

예상 출력: `(2, 3, 4) 24`.

### 4번

```python
import numpy as np

x = np.arange(5)
row = x[None, :]
column = x[:, None]
print(row.shape)
print(column.shape)
```

예상 출력은 `(1, 5)`, `(5, 1)`.

### 5번

```python
import numpy as np

x = np.array([[1.0, 10.0, 100.0], [3.0, 14.0, 104.0]])
column_mean = x.mean(axis=0, keepdims=True)
centered = x - column_mean
print(centered)
print(centered.mean(axis=0))
print(np.allclose(centered.mean(axis=0), 0.0))
```

예상 마지막 출력은 `True`다. `(1, 3)` 평균이 `(2, 3)`에 broadcasting된다.

### 6번

```python
import numpy as np

x = np.arange(24).reshape(2, 3, 4)
y = x.transpose(0, 2, 1)
print(x.shape)
print(y.shape)
```

예상 출력: `(2, 3, 4)`와 `(2, 4, 3)`.

### 7번

```python
import numpy as np

original = np.arange(6)
part = original[2:5].copy()
part[:] = -1
print(original)
print(part)
assert original.tolist() == [0, 1, 2, 3, 4, 5]
print("원본 보존")
```

예상 마지막 출력: `원본 보존`.

### 8번. 글자와 숫자 읽기

이번 배치에는 샘플이 32개이므로 B=32다. `[32,60,3]`은 샘플 32개, 샘플마다 60시점, 시점마다 특성 3개다. `[8,3,64,64]`는 RGB 이미지 8장, 채널 3개, 세로·가로 각각 64픽셀이다. H·W가 같아도 뜻까지 같지는 않다. 원본 시계열 표는 `[전체 시점 수,F]`일 수 있고, window를 만들고 배치로 묶은 뒤 `[B,T,F]`가 된다. B는 배치의 개수가 아니라 **배치 안의 샘플 수**다.

### 9번. shape뿐 아니라 값의 대응을 검사한다

```python
import torch

x = torch.arange(12).reshape(2, 3, 2)
y = x.permute(0, 2, 1)
assert tuple(y.shape) == (2, 2, 3)
assert y[1, 0, 2].item() == x[1, 2, 0].item() == 10
assert not torch.equal(y, x.reshape(2, 2, 3))
print(y.tolist())
```

정답은 `[[[0,2,4],[1,3,5]],[[6,8,10],[7,9,11]]]`이다. reshape는 이 예제에서 숫자를 읽는 순서대로 다시 묶기 때문에 특성별 시간 기록을 보존하지 못한다. 두 축의 크기가 우연히 같더라도 축의 의미가 다르면 교환은 여전히 필요할 수 있다.

### 10번. 오른쪽부터 비교하기

순서대로 `[4,3]`, **실패**, `[4,3]`, `[2,5,3]`이다. 두 번째는 마지막 축 3과 4가 맞지 않는다. 세 번째는 `[4,1]`과 `[1,3]`으로 맞춰 보면 양쪽이 확장된다. 네 번째는 시간 5개에 대한 값을 모든 배치와 특성에 적용한다.

```python
import torch

assert (torch.zeros(4, 3) + torch.zeros(3)).shape == (4, 3)
try:
    torch.zeros(4, 3) + torch.zeros(4)
except RuntimeError:
    print("마지막 축 3과 4가 달라 실패: 예상된 오류")
else:
    raise AssertionError("호환되지 않는 shape가 통과했습니다")
assert (torch.zeros(4, 1) + torch.zeros(3)).shape == (4, 3)
assert (torch.zeros(2, 5, 3) + torch.zeros(1, 5, 1)).shape == (2, 5, 3)
```

시점별 값 `[T]`를 `[B,T,F]`에 바로 더하면 T가 아니라 마지막 F와 비교된다. T=F일 때는 실행되어도 특성별 값처럼 적용된다. 실행 성공만으로 의도한 계산인지 판단하지 않는다.

### 11번. 오류 메시지 없이 잘못 계산되는 회귀 오차

그냥 빼면 `[[-1,-3],[1,-1]]`이고 MSE는 `(1+9+1+1)/4=3`이다. 샘플끼리 짝지으면 차이는 `[[-1],[-1]]`, MSE는 1이다.

```python
import torch

pred = torch.tensor([[2.], [4.]])
target = torch.tensor([3., 5.])
wrong = pred - target
correct = pred - target.unsqueeze(1)
assert tuple(wrong.shape) == (2, 2)
assert wrong.square().mean().item() == 3.0
assert tuple(correct.shape) == (2, 1)
assert correct.square().mean().item() == 1.0

# 마지막 배치에 샘플 하나만 남아도 [1]과 [1,1]을 명확히 구분한다.
one_pred = pred[:1]
one_target = target[:1].unsqueeze(1)
assert one_pred.shape == one_target.shape == (1, 1)
assert one_pred.squeeze(1).shape == (1,)
assert one_pred.squeeze().shape == ()  # 모든 크기 1 축이 사라져 scalar가 됨
```

샘플마다 연속값 하나를 예측하는 회귀 예제다. 다중출력 회귀에서는 예측과 정답을 둘 다 `[B,D]`로 맞춘다. 여기서 D는 예측할 연속값 정답 열의 개수이며, 앞에서 정의한 클래스 수 K와 다르다. 클래스 번호를 정답으로 사용하는 교차엔트로피 손실에서는 `[B,K]`와 `[B]`가 올바른 조합이므로 이 회귀 규칙을 그대로 적용하지 않는다.

### 12번. 채널별 평균

정답은 `[1,3,1,1]`이다. 배치·높이·너비에는 같은 값을 반복하고 채널에 따라 다른 평균을 뺀다. `[3]`을 그대로 쓰면 W=5와 비교되어 실패한다. W=3이면 실행될 수 있어 더 위험하다.

```python
import torch

images = torch.ones(2, 3, 4, 5)
mean = torch.tensor([0.1, 0.2, 0.3])
centered = images - mean.reshape(1, 3, 1, 1)
assert centered.shape == images.shape
assert torch.allclose(centered[0, :, 0, 0], torch.tensor([0.9, 0.8, 0.7]))
```

### 13번. 분류 출력과 RNN 출력은 다르다

클래스 번호는 `[1,0]`, shape는 `[2]`다. logits는 `[2,3]`이고 아직 확률이 아니다. RNN 입력은 `[2,5,3]`, sequence 출력은 `[2,5,4]`, 마지막 은닉 상태는 `[1,2,4]`다. 은닉 상태의 첫 1은 층 수 × 방향 수다.

```python
import torch

logits = torch.tensor([[0., 2., 1.], [3., 0., 1.]])
assert logits.argmax(dim=1).tolist() == [1, 0]

rnn = torch.nn.RNN(input_size=3, hidden_size=4, batch_first=True)
sequence_output, h_n = rnn(torch.zeros(2, 5, 3))
assert sequence_output.shape == (2, 5, 4)
assert h_n.shape == (1, 2, 4)

default_rnn = torch.nn.RNN(input_size=3, hidden_size=4)
default_output, _ = default_rnn(torch.zeros(5, 2, 3))  # [T,B,F]
assert default_output.shape == (5, 2, 4)
```

시점별 출력은 각 시점에서 계산한 은닉 상태다. 그 자체가 최종 클래스 점수는 아니다. 분류 문제에서는 문제 설계에 맞게 시점 선택·집계와 출력층을 추가한다.

</details>

## 완료 기준

- [ ] 2차원 배열에서 행과 열을 원하는 구간만 선택할 수 있다.
- [ ] `axis=0`과 `axis=1` 집계의 결과 shape를 실행 전에 맞힌다.
- [ ] reshape와 transpose를 말과 코드로 구분한다.
- [ ] broadcasting 규칙을 shape의 오른쪽부터 검사한다.
- [ ] 원본 보존이 필요할 때 `.copy()`를 명시한다.
- [ ] 기존 NumPy 실습 1–7번 중 6문제 이상을 정답 없이 풀었다.
- [ ] B·F·T·C·H·W·K와 전체 샘플 수 N을 구분한다.
- [ ] permute(0,2,1)의 숫자가 기존 축 번호임을 설명하고 값의 대응을 확인한다.
- [ ] RNN의 batch_first 조건과 logit/확률/클래스 번호를 구분한다.
- [ ] 추가 실습 8–13번을 풀고, 특히 회귀 문제에서 배열 크기 때문에 오류가 생기는 11번을 해설 없이 수정했다.

---

# 입문 7강. pandas로 표 불러오기와 행·열 선택

## 1. 왜 필요한가

NumPy 배열은 숫자 계산에 강하지만 "이 열은 속도, 저 열은 온도"라는 이름을 직접 관리하기 어렵다. pandas의 DataFrame은 행과 열에 이름이 있는 표다. CSV를 읽고, 데이터의 크기·자료형·결측을 살펴보고, 필요한 행과 열을 선택하는 일은 거의 모든 분석의 출발점이다.

이 강의를 마치면 다음을 할 수 있다.

- `Series`와 `DataFrame`의 차이를 설명한다.
- 딕셔너리와 CSV 문자열에서 DataFrame을 만든다.
- `head`, `shape`, `columns`, `dtypes`, `info`, `describe`로 데이터를 점검한다.
- 대괄호, `loc`, `iloc`로 행과 열을 선택한다.
- Boolean 조건으로 행을 필터링하고 원본을 보존한다.

## 2. Series와 DataFrame

`Series`는 index가 붙은 1차원 값이고, `DataFrame`은 여러 Series가 열 방향으로 모인 2차원 표라고 생각할 수 있다.

```python
import pandas as pd

temperatures = pd.Series(
    [21.5, 23.0, 19.8],
    index=["morning", "noon", "night"],
    name="temperature",
)

print(temperatures)
print("shape:", temperatures.shape)
print("dtype:", temperatures.dtype)
print("noon:", temperatures.loc["noon"])
```

예상 출력:

```text
morning    21.5
noon       23.0
night      19.8
Name: temperature, dtype: float64
shape: (3,)
dtype: float64
noon: 23.0
```

DataFrame은 열마다 자료형이 달라도 된다.

```python
import pandas as pd

df = pd.DataFrame(
    {
        "id": ["A01", "A02", "A03"],
        "speed": [52.1, 48.7, 60.2],
        "count": [10, 12, 9],
        "fault": [False, True, False],
    }
)

print(df)
print("shape:", df.shape)
print("columns:", df.columns.tolist())
```

예상 출력:

```text
    id  speed  count  fault
0  A01   52.1     10  False
1  A02   48.7     12   True
2  A03   60.2      9  False
shape: (3, 4)
columns: ['id', 'speed', 'count', 'fault']
```

shape `(3, 4)`는 "3행, 4열"이다. 머신러닝 관점에서는 흔히 행이 샘플, 열이 특성이다.

## 3. CSV를 불러오는 원리

CSV는 값이 쉼표로 구분된 텍스트 파일이다. 외부 파일 없이 연습하기 위해 `StringIO`로 문자열을 파일처럼 읽는다.

```python
from io import StringIO
import pandas as pd

csv_text = """id,speed,temperature,label
A01,52.1,23.5,normal
A02,48.7,25.0,fault
A03,60.2,24.1,normal
"""

df = pd.read_csv(StringIO(csv_text))
print(df)
print(df.shape)
```

예상 출력:

```text
    id  speed  temperature   label
0  A01   52.1         23.5  normal
1  A02   48.7         25.0   fault
2  A03   60.2         24.1  normal
(3, 4)
```

실제 파일이라면 `pd.read_csv("train.csv")`처럼 읽는다. 처음에는 다음 항목을 확인한다.

- 구분자가 쉼표가 아니면 `sep="\t"` 등 지정
- 한글 파일에서 문제가 있으면 먼저 공식 제공 인코딩 확인
- 결측 표기가 `?`, `NA`, `missing`이면 `na_values=[...]` 고려
- 날짜 열은 먼저 문자열로 읽고 명시적으로 변환하는 편이 안전
- ID처럼 앞의 0을 보존해야 하는 열은 `dtype={"id": "string"}` 지정

아래 예제는 문자열 `NA`와 `?`를 결측으로 읽고 id를 문자열 dtype으로 유지한다.

```python
from io import StringIO
import pandas as pd

csv_text = """id,speed,temperature
001,52.1,23.5
002,NA,25.0
003,60.2,?
"""

df = pd.read_csv(
    StringIO(csv_text),
    dtype={"id": "string"},
    na_values=["NA", "?"],
)

print(df)
print(df.dtypes)
```

예상 결과에서 id의 `001`, `002`, `003` 앞자리 0이 유지되고 speed와 temperature 결측이 `NaN`으로 보인다. pandas 버전에 따라 문자열 dtype의 세부 표시 방식은 다를 수 있다.

## 4. 데이터를 받으면 먼저 하는 7가지 점검

```python
from io import StringIO
import pandas as pd

csv_text = """id,speed,temperature,label
A01,52.1,23.5,normal
A02,48.7,,fault
A03,60.2,24.1,normal
A04,55.0,24.8,fault
"""
df = pd.read_csv(StringIO(csv_text))

print("1) shape:", df.shape)
print("2) columns:", df.columns.tolist())
print("3) head:\n", df.head(2))
print("4) dtypes:\n", df.dtypes)
print("5) 결측 수:\n", df.isna().sum())
print("6) label 빈도:\n", df["label"].value_counts(dropna=False))
numeric_df = df.select_dtypes(include="number")
print("7) 수치 요약:\n", numeric_df.describe())
```

핵심 예상 결과:

```text
1) shape: (4, 4)
2) columns: ['id', 'speed', 'temperature', 'label']
temperature 결측 수: 1
label은 normal 2개, fault 2개
```

`df.info()`는 열마다 결측값이 아닌 값의 개수와 자료형, 메모리 사용량을 직접 출력한다. 반환값은 `None`이므로 `result = df.info()`에 분석 결과가 저장되는 것은 아니다.

```python
import pandas as pd

df = pd.DataFrame({"x": [1, 2, 3], "name": ["a", "b", "c"]})
returned = df.info()
print("info 반환값:", returned)
```

예상 출력에는 DataFrame 정보가 먼저 나오고 마지막 줄은 `info 반환값: None`이다.

## 5. 열 선택: Series와 DataFrame 결과를 구분

```python
import pandas as pd

df = pd.DataFrame(
    {
        "id": ["A", "B", "C"],
        "speed": [50.0, 60.0, 55.0],
        "temp": [20.0, 22.0, 21.0],
    }
)

one_column = df["speed"]
one_column_table = df[["speed"]]
two_columns = df[["speed", "temp"]]

print(type(one_column), one_column.shape)
print(type(one_column_table), one_column_table.shape)
print(two_columns)
```

예상 출력의 핵심:

```text
<class 'pandas.core.series.Series'> (3,)
<class 'pandas.core.frame.DataFrame'> (3, 1)
   speed  temp
0   50.0  20.0
1   60.0  22.0
2   55.0  21.0
```

대괄호 하나에 열 이름 문자열을 주면 Series, 열 이름 목록을 주면 DataFrame이다. 모델 입력 `X`를 2차원으로 유지할 때 `df[["speed"]]`가 필요할 수 있다.

## 6. loc와 iloc

- `loc[행 label, 열 label]`: 이름 기반, slice 끝 label을 포함
- `iloc[행 위치, 열 위치]`: 정수 위치 기반, Python slice처럼 끝 위치 제외

```python
import pandas as pd

df = pd.DataFrame(
    {
        "speed": [50, 60, 55],
        "temp": [20, 22, 21],
        "label": [0, 1, 0],
    },
    index=["A01", "A02", "A03"],
)

print("loc 한 값:", df.loc["A02", "temp"])
print("iloc 한 값:", df.iloc[1, 1])
print("loc 구간:\n", df.loc["A01":"A02", ["speed", "temp"]])
print("iloc 구간:\n", df.iloc[0:2, 0:2])
```

예상 출력에서 한 값은 둘 다 `22`이고, 두 구간은 A01과 A02의 speed·temp를 담은 `(2, 2)` 표다. `loc["A01":"A02"]`는 A02를 포함하지만 `iloc[0:2]`는 위치 2를 제외한다.

index가 0, 1, 2 같은 정수여도 `loc[1]`은 label 1, `iloc[1]`은 두 번째 위치라는 의미다. 필터 후 index가 불연속이면 둘은 쉽게 달라진다.

```python
import pandas as pd

df = pd.DataFrame({"value": [10, 20, 30]}, index=[100, 200, 300])
print("label 200:", df.loc[200, "value"])
print("두 번째 위치:", df.iloc[1, 0])
```

예상 출력은 둘 다 20이지만 접근 기준이 다르다.

## 7. 행 필터링과 정렬

```python
import pandas as pd

df = pd.DataFrame(
    {
        "id": ["A", "B", "C", "D"],
        "speed": [45, 62, 58, 70],
        "temp": [21, 25, 23, 28],
    }
)

mask = (df["speed"] >= 55) & (df["temp"] < 27)
selected = df.loc[mask, ["id", "speed"]].copy()
ordered = selected.sort_values("speed", ascending=False)

print(mask.tolist())
print(ordered)
```

예상 출력:

```text
[False, True, True, False]
  id  speed
1  B     62
2  C     58
```

pandas Series의 여러 조건도 각 조건을 괄호로 감싸고 `&`, `|`, `~`를 사용한다. `and`나 `or`를 쓰면 "Series 전체를 하나의 참·거짓 값으로 판단할 수 없다"라는 오류가 난다.

문자열 조건도 가능하다.

```python
import pandas as pd

df = pd.DataFrame(
    {"city": ["Seoul", "Busan", "Seoul", "Daegu"], "score": [80, 90, 70, 85]}
)

selected = df.loc[df["city"].isin(["Seoul", "Daegu"])]
print(selected.reset_index(drop=True))
```

예상 출력은 Seoul 두 행과 Daegu 한 행을 포함한다. `reset_index(drop=True)`는 선택 후 남은 기존 index를 버리고 0부터 다시 붙인다.

## 8. 열 만들기·이름 바꾸기·삭제

```python
import pandas as pd

df = pd.DataFrame({"distance_km": [10.0, 20.0], "hours": [0.5, 1.0]})
out = df.copy()

out["speed_kmh"] = out["distance_km"] / out["hours"]
out = out.rename(columns={"hours": "duration_hours"})
out = out.drop(columns=["distance_km"])

print("원본 열:", df.columns.tolist())
print(out)
```

예상 출력:

```text
원본 열: ['distance_km', 'hours']
   duration_hours  speed_kmh
0             0.5       20.0
1             1.0       20.0
```

`out = df.copy()`로 시작했기 때문에 원본은 유지된다. 시험 함수에서 "입력 DataFrame을 변경하지 말라"는 명세가 있으면 특히 중요하다.

## 9. 흔한 오류와 해결법

| 오류·실수 | 원인 | 해결 |
|---|---|---|
| `KeyError: 'speed'` | 열 이름 오타·공백 | `df.columns.tolist()` 출력 |
| `df["x"]`가 2차원이라고 생각 | Series `(B,)` 반환 | 2차원이면 `df[["x"]]` |
| `loc`와 `iloc` 혼동 | label과 위치를 섞음 | index 출력 후 기준 명시 |
| 조건에서 `and`, `or` 사용 | Series 전체의 참/거짓 불명확 | 괄호 + `&`, `|` |
| `df.info()` 결과를 변수로 분석 | 반환값은 `None` | 화면 출력으로 확인 |
| 필터한 표를 수정해 경고·모호한 동작 | view/copy 불분명 | 수정할 결과에 `.copy()` |
| ID의 앞자리 0 소실 | 숫자로 자동 추론 | 읽을 때 string dtype 지정 |

## 10. 실습문제

1. 이름, 나이, 점수 세 열과 4행을 가진 DataFrame을 딕셔너리로 만들고 shape와 dtypes를 출력하라.
2. 제공된 CSV 문자열에서 `id`, `x`, `y`를 읽고 첫 두 행과 열 이름을 출력하라.
3. 한 열을 Series와 `(B, 1)` DataFrame으로 각각 선택해 배열 크기의 차이를 출력하라.
4. 점수가 80 이상이고 나이가 30 미만인 행의 이름과 점수만 선택하라.
5. `loc`로 label `s2` 행의 `x`, `y`를 선택하고, `iloc`로 첫 두 행·첫 두 열을 선택하라.
6. 원본을 보존하면서 `x + y`인 `total` 열을 추가한 새 DataFrame을 만들라.
7. 점수를 내림차순 정렬하고 index를 0부터 다시 붙여라.

<details>
<summary>입문 7강 정답·해설 펼치기</summary>

### 1번

```python
import pandas as pd

df = pd.DataFrame(
    {
        "name": ["A", "B", "C", "D"],
        "age": [24, 31, 28, 35],
        "score": [88.0, 75.0, 91.0, 82.0],
    }
)
print(df)
print(df.shape)
print(df.dtypes)
```

예상 shape는 `(4, 3)`이다.

### 2번

```python
from io import StringIO
import pandas as pd

csv_text = """id,x,y
A,1,10
B,2,20
C,3,30
"""
df = pd.read_csv(StringIO(csv_text))
print(df.head(2))
print(df.columns.tolist())
```

예상 열 목록: `['id', 'x', 'y']`.

### 3번

```python
import pandas as pd

df = pd.DataFrame({"score": [80, 90, 70]})
series = df["score"]
table = df[["score"]]
print(series.shape)
print(table.shape)
```

예상 출력: `(3,)`, `(3, 1)`.

### 4번

```python
import pandas as pd

df = pd.DataFrame(
    {"name": ["A", "B", "C"], "age": [25, 32, 28], "score": [85, 90, 75]}
)
mask = (df["score"] >= 80) & (df["age"] < 30)
print(df.loc[mask, ["name", "score"]])
```

예상 결과는 A 한 행이다.

### 5번

```python
import pandas as pd

df = pd.DataFrame(
    {"x": [1, 2, 3], "y": [10, 20, 30], "z": [100, 200, 300]},
    index=["s1", "s2", "s3"],
)
print(df.loc["s2", ["x", "y"]])
print(df.iloc[:2, :2])
```

`loc` 결과의 x=2, y=20이고 `iloc` 결과는 s1·s2의 x·y 표다.

### 6번

```python
import pandas as pd

df = pd.DataFrame({"x": [1, 2], "y": [10, 20]})
out = df.copy()
out["total"] = out["x"] + out["y"]
print(df.columns.tolist())
print(out)
```

원본 열은 여전히 `['x', 'y']`이고 out에만 total이 있다.

### 7번

```python
import pandas as pd

df = pd.DataFrame({"name": ["A", "B", "C"], "score": [75, 92, 85]})
ordered = df.sort_values("score", ascending=False).reset_index(drop=True)
print(ordered)
```

예상 이름 순서는 B, C, A이고 index는 0, 1, 2다.

</details>

## 완료 기준

- [ ] Series `(B,)`와 한 열짜리 DataFrame `(B, 1)`을 구분한다.
- [ ] 새 데이터를 받으면 shape·columns·dtypes·결측·분포를 점검한다.
- [ ] `loc`와 `iloc`의 기준 차이를 설명한다.
- [ ] 여러 조건을 괄호와 `&`, `|`로 조합할 수 있다.
- [ ] 입력 표를 보존해야 할 때 `.copy()`로 새 표를 만든다.
- [ ] 실습 7문제 중 6문제 이상을 정답 없이 풀었다.

---

# 입문 8강. pandas로 결측치 처리, 그룹별 집계, 표 병합

## 1. 왜 필요한가

실제 데이터는 비어 있는 값, 숫자처럼 보이는 문자열, 중복 행, 서로 다른 표에 나뉜 정보를 포함한다. 모델은 이러한 문제를 자동으로 이해하지 못한다. 정제 규칙을 명시하고, 학습 데이터에서 계산한 통계만 검증·평가 데이터에 적용하며, 표를 합친 뒤 행 수가 예상과 같은지 확인해야 데이터 누수를 피할 수 있다.

이 강의를 마치면 다음을 할 수 있다.

- 결측 개수와 비율을 확인하고 목적에 맞게 제거·대체한다.
- `to_numeric`, `astype`, `to_datetime`으로 자료형을 변환한다.
- 중복 행과 범주 빈도를 점검한다.
- `groupby`로 그룹별 집계를 만든다.
- `merge`와 `concat`의 차이를 알고 결합 결과를 검증한다.

## 2. 결측값 찾기

pandas에서 수치 결측은 흔히 `NaN`으로 보인다. 값이 없는지 비교 연산 `== np.nan`으로 검사하지 말고 `isna()`를 사용한다.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame(
    {
        "speed": [50.0, np.nan, 60.0, 55.0],
        "temp": [20.0, 22.0, np.nan, 24.0],
        "label": ["normal", "fault", "normal", None],
    }
)

print(df)
print("열별 결측 수:\n", df.isna().sum())
print("열별 결측 비율:\n", df.isna().mean())
print("행별 결측 수:", df.isna().sum(axis=1).tolist())
```

예상 핵심 결과:

```text
speed 결측 1, temp 결측 1, label 결측 1
각 열 결측 비율 0.25
행별 결측 수 [0, 1, 1, 1]
```

`NaN == NaN`은 거짓이므로 전용 검사를 쓴다.

```python
import numpy as np
import pandas as pd

value = np.nan
print(value == np.nan)
print(pd.isna(value))
```

예상 출력:

```text
False
True
```

## 3. 결측값 제거와 대체

무조건 행을 지우거나 무조건 평균으로 채우는 규칙은 없다. 열의 의미, 결측 원인, 결측 비율, 모델을 함께 고려한다.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame(
    {
        "speed": [50.0, np.nan, 60.0, 55.0],
        "temp": [20.0, 22.0, np.nan, 24.0],
        "label": ["normal", "fault", "normal", None],
    }
)

numeric_filled = df.copy()
numeric_filled["speed"] = numeric_filled["speed"].fillna(df["speed"].median())
numeric_filled["temp"] = numeric_filled["temp"].fillna(df["temp"].median())
numeric_filled["label"] = numeric_filled["label"].fillna("unknown")

complete_label = df.dropna(subset=["label"]).copy()

print(numeric_filled)
print("label이 있는 행 수:", len(complete_label))
```

예상 결과:

```text
speed의 NaN은 중앙값 55.0
temp의 NaN은 중앙값 22.0
label의 None은 unknown
label이 있는 행 수: 3
```

### 중요한 원칙: 결측치를 채울 값은 학습 데이터에서만 계산한다

검증·평가 데이터까지 포함해 중앙값을 구하면, 학습에 사용해서는 안 되는 정보가 전처리에 반영될 수 있다.

```python
import numpy as np
import pandas as pd

train = pd.DataFrame({"speed": [40.0, 50.0, np.nan, 60.0]})
valid = pd.DataFrame({"speed": [np.nan, 1000.0]})

train_median = train["speed"].median()
train_out = train.copy()
valid_out = valid.copy()
train_out["speed"] = train_out["speed"].fillna(train_median)
valid_out["speed"] = valid_out["speed"].fillna(train_median)

print("train 중앙값:", train_median)
print("valid 정제:\n", valid_out)
```

예상 출력:

```text
train 중앙값: 50.0
valid 정제:
     speed
0    50.0
1  1000.0
```

## 4. 숫자와 날짜 자료형 정리

`astype(float)`는 하나라도 잘못된 문자열이 있으면 오류가 난다. 정제 단계에서는 `pd.to_numeric(..., errors="coerce")`로 바꿀 수 없는 값을 결측으로 표시한 뒤 개수를 점검하는 방법이 유용하다.

```python
import pandas as pd

df = pd.DataFrame(
    {
        "raw_speed": ["50.1", "bad", "62.0"],
        "timestamp": ["2026-01-01 09:00", "2026-01-02 10:30", "invalid"],
    }
)

out = df.copy()
out["speed"] = pd.to_numeric(out["raw_speed"], errors="coerce")
out["time"] = pd.to_datetime(out["timestamp"], errors="coerce")

print(out)
print("speed 변환 실패:", out["speed"].isna().sum())
print("time 변환 실패:", out["time"].isna().sum())
```

예상 결과에서 `bad`의 speed와 `invalid`의 time이 결측(`NaN`, `NaT`)이 되고 각각 변환 실패 수는 1이다. 변환 실패를 조용히 채우기 전에 원본 값이 왜 잘못되었는지 확인한다.

범주형 문자열의 앞뒤 공백과 대소문자도 정리할 수 있다.

```python
import pandas as pd

df = pd.DataFrame({"label": [" Normal", "FAULT ", "normal", None]})
out = df.copy()
out["label_clean"] = out["label"].astype("string").str.strip().str.lower()

print(out)
print(out["label_clean"].value_counts(dropna=False))
```

예상 결과에서 `Normal`과 `normal`은 모두 `normal`, `FAULT`는 `fault`가 된다. 결측은 문자열 `"None"`으로 바꾸지 않고 결측 상태를 유지한다.

## 5. 중복과 고유값

```python
import pandas as pd

df = pd.DataFrame(
    {
        "id": ["A", "A", "B", "B"],
        "time": [1, 1, 1, 2],
        "value": [10, 10, 20, 21],
    }
)

print("완전 중복 수:", df.duplicated().sum())
print("id-time 중복 수:", df.duplicated(subset=["id", "time"]).sum())

deduplicated = df.drop_duplicates(subset=["id", "time"], keep="last")
print(deduplicated.reset_index(drop=True))
```

예상 출력:

```text
완전 중복 수: 1
id-time 중복 수: 1
  id  time  value
0  A     1     10
1  B     1     20
2  B     2     21
```

중복을 제거하기 전에 어떤 열 조합이 한 행을 유일하게 만들어야 하는지 정의한다. 같은 id라도 시간이 다르면 정상적인 별도 관측일 수 있다.

## 6. groupby로 그룹별 요약

`groupby`는 보통 <strong>나누기(split) → 계산(apply) → 합치기(combine)</strong>로 생각한다.

```python
import pandas as pd

df = pd.DataFrame(
    {
        "machine": ["A", "A", "B", "B", "B"],
        "speed": [50, 54, 60, 66, 63],
        "fault": [0, 1, 0, 1, 1],
    }
)

summary = (
    df.groupby("machine", as_index=False)
    .agg(
        sample_count=("speed", "size"),
        mean_speed=("speed", "mean"),
        fault_rate=("fault", "mean"),
    )
)

print(summary)
```

예상 출력:

```text
  machine  sample_count  mean_speed  fault_rate
0       A             2        52.0    0.500000
1       B             3        63.0    0.666667
```

`size`는 행 수, `count`는 해당 열의 결측이 아닌 값 수라는 차이가 있다.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"group": ["A", "A", "B"], "value": [1.0, np.nan, 3.0]})
summary = df.groupby("group")["value"].agg(["size", "count"])
print(summary)
```

예상 출력:

```text
       size  count
group             
A         2      1
B         1      1
```

## 7. merge: key를 기준으로 열 결합

두 표에서 공통으로 사용하는 키, 예를 들어 장비 ID가 같은 행을 찾아 정보를 합친다. SQL의 조인(join)과 같은 방식이다.

```python
import pandas as pd

measurements = pd.DataFrame(
    {"machine_id": ["M1", "M2", "M3"], "speed": [50, 60, 55]}
)
machines = pd.DataFrame(
    {"machine_id": ["M1", "M2", "M4"], "factory": ["A", "B", "C"]}
)

left_joined = measurements.merge(
    machines,
    on="machine_id",
    how="left",
    validate="one_to_one",
    indicator=True,
)

print(left_joined)
```

예상 출력:

```text
  machine_id  speed factory     _merge
0          M1     50       A       both
1          M2     60       B       both
2          M3     55     NaN  left_only
```

- `inner`: 양쪽에 key가 있는 행만
- `left`: 왼쪽 행을 모두 유지
- `right`: 오른쪽 행을 모두 유지
- `outer`: 어느 한쪽에라도 있는 key 모두

`validate="one_to_one"`은 두 표 모두에서 키가 중복되지 않는지 검사한다. 다대일 결합이면 `many_to_one`을 쓸 수 있다. 결합 전후 행 수와 key 중복을 확인하지 않으면 결과의 행 수가 예상보다 크게 늘어날 수 있다.

```python
import pandas as pd

events = pd.DataFrame({"machine_id": ["M1", "M1", "M2"], "value": [1, 2, 3]})
meta = pd.DataFrame({"machine_id": ["M1", "M2"], "factory": ["A", "B"]})

joined = events.merge(meta, on="machine_id", how="left", validate="many_to_one")
assert len(joined) == len(events)
print(joined)
```

예상 결과는 3행이고 M1 두 행 모두 factory A가 붙는다.

## 8. concat: 같은 구조의 표 이어 붙이기

`concat`은 key를 찾아 붙이는 merge와 달리 축을 따라 표를 이어 붙인다.

```python
import pandas as pd

january = pd.DataFrame({"id": ["A", "B"], "value": [10, 20]})
february = pd.DataFrame({"id": ["C"], "value": [30]})

rows = pd.concat([january, february], axis=0, ignore_index=True)
print(rows)
```

예상 출력:

```text
  id  value
0  A     10
1  B     20
2  C     30
```

열 방향 concat은 index를 기준으로 맞춘다.

```python
import pandas as pd

features = pd.DataFrame({"x1": [1, 2]}, index=["A", "B"])
extra = pd.DataFrame({"x2": [10, 20]}, index=["A", "B"])

columns = pd.concat([features, extra], axis=1)
print(columns)
```

예상 출력:

```text
   x1  x2
A   1  10
B   2  20
```

index가 다르면 행 위치가 아니라 label에 맞춰지고 결측이 생길 수 있다. 단순히 옆에 붙이고 싶어도 index가 의미 있는지 먼저 확인한다.

## 9. 정제 파이프라인 예제

한 번에 여러 변경을 하더라도 단계별 검사를 남긴다.

```python
from io import StringIO
import pandas as pd

csv_text = """id,group,value,label
A1, X ,10.0,normal
A2,X,bad,FAULT
A2,X,bad,FAULT
B1,Y,30.0,normal
"""

raw = pd.read_csv(StringIO(csv_text), dtype={"id": "string"})
clean = raw.copy()
clean["group"] = clean["group"].astype("string").str.strip()
clean["label"] = clean["label"].astype("string").str.lower()
clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
clean = clean.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

median = clean["value"].median()
clean["value"] = clean["value"].fillna(median)

assert clean["id"].is_unique
assert clean["value"].notna().all()

print(clean)
print(clean.dtypes)
```

예상 결과는 A1, A2, B1 세 행이고 A2의 value는 유효값 10과 30의 중앙값 20으로 채워진다. label은 소문자가 된다.

## 10. 흔한 오류와 해결법

| 오류·실수 | 원인 | 해결 |
|---|---|---|
| `x == np.nan`으로 결측 검사 | NaN은 자기 자신과도 같지 않음 | `pd.isna`, `.isna()` |
| 전체 데이터 중앙값으로 validation 채움 | validation 정보 누수 | train에서 통계 계산 후 적용 |
| `errors="coerce"` 후 바로 fill | 변환 실패 원인을 못 봄 | 결측 개수와 원본 이상값 점검 |
| `groupby().count()`를 행 수로 착각 | 결측은 count에서 제외 | 행 수면 `size` |
| merge 후 행이 폭증 | key가 양쪽에서 중복 | 중복 검사, `validate` 사용 |
| concat axis=1 후 NaN | index label이 다름 | index 정렬 의도 확인 |
| 중복 제거 기준을 모든 열로 둠 | 업무상 key 중복을 놓침 | 고유해야 할 key subset 정의 |

## 11. 실습문제

1. 각 열의 결측 수와 비율을 하나의 DataFrame으로 만들어라.
2. 문자열 `['10', 'bad', '30']`을 숫자로 변환하고 실패 개수를 출력한 뒤 중앙값으로 채워라.
3. `city` 문자열의 앞뒤 공백을 없애고 소문자로 통일한 뒤 빈도를 구하라.
4. 그룹별 행 수, value 열의 평균, target 열의 평균을 구하라. 이름 있는 집계(named aggregation)를 사용해 결과 열에 이름을 붙여라.
5. 여러 이벤트가 있는 왼쪽 표와 machine별 한 행인 메타 표를 `many_to_one`으로 left merge하고 행 수가 유지되는지 assert하라.
6. 같은 열 구조의 두 DataFrame을 행 방향으로 합치고 index를 0부터 다시 붙여라.
7. id와 time 조합의 중복 수를 확인하고 마지막 행을 남겨 중복을 제거하라.

<details>
<summary>입문 8강 정답·해설 펼치기</summary>

### 1번

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [None, "x", "y"]})
report = pd.DataFrame(
    {"missing_count": df.isna().sum(), "missing_rate": df.isna().mean()}
)
print(report)
```

예상 결과는 a와 b 모두 결측 수 1, 비율 약 0.3333이다.

### 2번

```python
import pandas as pd

s = pd.Series(["10", "bad", "30"])
numbers = pd.to_numeric(s, errors="coerce")
print("실패:", numbers.isna().sum())
numbers = numbers.fillna(numbers.median())
print(numbers.tolist())
```

예상 출력은 실패 1, 값 `[10.0, 20.0, 30.0]`.

### 3번

```python
import pandas as pd

df = pd.DataFrame({"city": [" Seoul", "BUSAN ", "seoul", None]})
clean = df["city"].astype("string").str.strip().str.lower()
print(clean.value_counts(dropna=False))
```

예상 빈도는 seoul 2, busan 1, 결측 1이다.

### 4번

```python
import pandas as pd

df = pd.DataFrame(
    {"group": ["A", "A", "B"], "value": [10.0, 20.0, 30.0], "target": [0, 1, 1]}
)
summary = df.groupby("group", as_index=False).agg(
    rows=("value", "size"),
    mean_value=("value", "mean"),
    target_rate=("target", "mean"),
)
print(summary)
```

예상 A는 rows 2, mean 15, target_rate 0.5이고 B는 1, 30, 1.0이다.

### 5번

```python
import pandas as pd

events = pd.DataFrame({"machine": ["M1", "M1", "M2"], "value": [1, 2, 3]})
meta = pd.DataFrame({"machine": ["M1", "M2"], "factory": ["A", "B"]})
joined = events.merge(meta, on="machine", how="left", validate="many_to_one")
assert len(joined) == len(events)
print(joined)
```

예상 결과는 3행이며 factory 결측이 없다.

### 6번

```python
import pandas as pd

first = pd.DataFrame({"id": ["A", "B"], "x": [1, 2]})
second = pd.DataFrame({"id": ["C"], "x": [3]})
combined = pd.concat([first, second], axis=0, ignore_index=True)
print(combined)
```

예상 index는 0, 1, 2이고 id는 A, B, C다.

### 7번

```python
import pandas as pd

df = pd.DataFrame(
    {"id": ["A", "A", "A", "B"], "time": [1, 1, 2, 1], "value": [10, 11, 12, 20]}
)
print("중복:", df.duplicated(subset=["id", "time"]).sum())
out = df.drop_duplicates(subset=["id", "time"], keep="last").reset_index(drop=True)
print(out)
```

예상 중복 수는 1이고 A-time1의 value는 마지막 값 11이 남는다.

</details>

## 완료 기준

- [ ] 결측 수와 비율을 열별로 계산할 수 있다.
- [ ] 숫자·날짜 변환 실패를 결측으로 표시하고 실패 수를 확인한다.
- [ ] 학습 데이터에서 구한 통계값을 검증·테스트 데이터에 그대로 적용해야 하는 이유를 설명한다.
- [ ] groupby의 `size`와 `count` 차이를 안다.
- [ ] merge와 concat을 구분하고 merge 관계를 `validate`로 검사한다.
- [ ] 실습 7문제 중 6문제 이상을 정답 없이 풀었다.

---

# 입문 9강. Matplotlib로 데이터 탐색과 그래프 그리기

## 1. 왜 필요한가

EDA(Exploratory Data Analysis, 탐색적 데이터 분석)는 모델을 만들기 전에 데이터가 어떤 모양인지 질문하고 확인하는 과정이다. 평균 하나만 보면 분포의 치우침, 이상치, 그룹 차이, 두 변수의 관계를 놓칠 수 있다. 표의 요약값과 그래프를 함께 보되, 예쁜 그림보다 **검증할 질문과 축의 의미**가 먼저다.

이 강의를 마치면 다음을 할 수 있다.

- `Figure`와 `Axes`의 역할을 설명한다.
- 꺾은선그래프·막대그래프·히스토그램·산점도·상자그림을 목적에 맞게 그린다.
- 제목·축 이름·범례·격자를 추가하고 그림을 저장한다.
- 결측·이상치·target 분포·변수 관계를 순서대로 점검한다.
- 그래프에서 본 패턴을 숫자 요약으로 다시 검증한다.

## 2. Figure와 Axes

Matplotlib에서 `Figure`는 전체 도화지, `Axes`는 실제 그래프 하나가 그려지는 영역이다. 처음부터 객체 지향 방식인 `fig, ax = plt.subplots()`를 익히면 여러 그래프를 다루기 쉽다.

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [2, 4, 3, 6]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, marker="o", label="measurement")
ax.set_title("Simple line plot")
ax.set_xlabel("time")
ax.set_ylabel("value")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
```

예상 출력: 가로축 time 1~4, 세로축 value인 꺾은선 그래프가 나타난다. 점에는 원형 marker가 있고 범례 `measurement`가 보인다. 노트북에서는 셀 아래에 나타나며 일반 Python 환경에서는 별도 창이 열릴 수 있다.

### 어떤 그래프를 언제 쓰나

| 질문 | 적합한 그래프 | 예시 |
|---|---|---|
| 시간에 따라 어떻게 변하는가? | line | 시간별 온도 |
| 범주별 크기는 어떤가? | bar | label 개수 |
| 한 수치 변수의 분포는? | histogram, boxplot | 속도 분포·이상치 |
| 두 수치 변수 관계는? | scatter | 온도와 진동 |
| 여러 그래프를 비교하고 싶은가? | subplots | class별 분포 비교 |

## 3. 범주 개수: 막대그래프

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({"label": ["normal", "fault", "normal", "warning", "normal", "fault"]})
counts = df["label"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(counts.index, counts.values, color=["#d95f5f", "#5f83c5", "#e0a84f"])
ax.set_title("Label counts")
ax.set_xlabel("label")
ax.set_ylabel("count")
ax.set_ylim(0, counts.max() + 1)

for position, count in enumerate(counts.values):
    ax.text(position, count + 0.05, str(count), ha="center")

fig.tight_layout()
plt.show()

print(counts.to_dict())
```

예상 텍스트 출력:

```text
{'fault': 2, 'normal': 3, 'warning': 1}
```

예상 그래프: 세 범주의 막대 높이가 각각 2, 3, 1이다. 분류 문제에서는 각 정답 클래스가 몇 번 나타나는지 먼저 살펴본다. 클래스별 개수가 크게 다르면 정확도만으로는 성능을 충분히 평가하지 못할 수 있다.

## 4. 수치 데이터의 분포: 히스토그램과 상자그림

히스토그램은 값의 범위를 여러 구간(bin)으로 나누고, 구간마다 값이 몇 개 있는지 보여 준다. 구간 수에 따라 분포가 다르게 보일 수 있으므로 구간 수를 바꿔 보고 요약 통계량도 함께 확인한다.

```python
import matplotlib.pyplot as plt
import numpy as np

values = np.array([48, 50, 51, 52, 52, 53, 54, 55, 56, 80], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(values, bins=5, edgecolor="black", color="#5f83c5")
axes[0].set_title("Histogram")
axes[0].set_xlabel("speed")
axes[0].set_ylabel("count")

axes[1].boxplot(values, vert=True)
axes[1].set_title("Boxplot")
axes[1].set_ylabel("speed")

fig.tight_layout()
plt.show()

print("mean:", values.mean())
print("median:", np.median(values))
print("min/max:", values.min(), values.max())
```

예상 텍스트 출력:

```text
mean: 55.1
median: 52.5
min/max: 48.0 80.0
```

80이라는 큰 값 때문에 평균이 중앙값보다 올라간다. boxplot에서 80이 멀리 떨어져 보일 수 있다. 하지만 "그래프에서 멀다"는 이유만으로 삭제하지 않는다. 센서 오류인지 실제 고장 신호인지 업무 의미를 확인한다.

## 5. 두 수치 변수의 관계: 산점도

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(
    {
        "temperature": [20, 21, 22, 24, 25, 27, 29, 30],
        "vibration": [0.10, 0.12, 0.15, 0.20, 0.23, 0.31, 0.42, 0.50],
        "label": [0, 0, 0, 0, 1, 1, 1, 1],
    }
)

fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(
    df["temperature"],
    df["vibration"],
    c=df["label"],
    cmap="coolwarm",
    s=70,
    edgecolor="black",
)
ax.set_title("Temperature vs vibration")
ax.set_xlabel("temperature")
ax.set_ylabel("vibration")
legend = ax.legend(*scatter.legend_elements(), title="label")
ax.add_artist(legend)
ax.grid(alpha=0.2)
fig.tight_layout()
plt.show()

print(df[["temperature", "vibration"]].corr())
```

예상 그래프: 온도가 올라갈수록 진동도 올라가는 양의 관계가 보이고 label 0과 1 색이 어느 정도 나뉜다. 예상 상관계수는 0에 가까운 값이 아니라 강한 양수다.

상관은 인과관계를 증명하지 않는다. 또 Pearson 상관이 낮아도 곡선 관계가 있을 수 있고, 한 이상치가 상관을 크게 바꿀 수 있다. 산점도와 상관계수를 함께 살펴본다.

## 6. 그룹별 분포 비교

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(
    {
        "label": ["normal"] * 6 + ["fault"] * 6,
        "vibration": [0.10, 0.13, 0.12, 0.18, 0.15, 0.16, 0.45, 0.52, 0.48, 0.60, 0.55, 0.50],
    }
)

normal = df.loc[df["label"] == "normal", "vibration"]
fault = df.loc[df["label"] == "fault", "vibration"]

fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([normal, fault])
ax.set_xticks([1, 2])
ax.set_xticklabels(["normal", "fault"])
ax.set_title("Vibration by label")
ax.set_xlabel("label")
ax.set_ylabel("vibration")
fig.tight_layout()
plt.show()

print(df.groupby("label")["vibration"].agg(["count", "mean", "median"]))
```

예상 결과: fault 그룹의 진동 중앙값·평균이 normal 그룹보다 높고 boxplot도 위쪽에 위치한다. boxplot을 그린 뒤 눈금 위치와 문자열을 따로 지정해 버전에 따른 인자 이름 차이를 피했다.

## 7. 여러 열을 한 화면에서 점검

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(
    {
        "speed": [45, 48, 50, 52, 60, 65, 70, 75],
        "temperature": [20, 21, 21, 23, 26, 28, 30, 32],
        "vibration": [0.10, 0.11, 0.13, 0.15, 0.30, 0.40, 0.52, 0.65],
    }
)

columns = ["speed", "temperature", "vibration"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

for ax, column in zip(axes, columns):
    ax.hist(df[column], bins=4, edgecolor="black")
    ax.set_title(column)
    ax.set_xlabel("value")
    ax.set_ylabel("count")

fig.suptitle("Numeric feature distributions")
fig.tight_layout()
plt.show()
```

예상 출력: 세 수치 열의 histogram이 한 행에 나타난다. `axes`도 배열이므로 Python 반복문과 `zip`으로 각 subplot을 설정할 수 있다.

## 8. 그림 저장과 자원 정리

아래 코드는 임시 폴더에 PNG를 저장하므로 외부 데이터가 없어도 독립 실행된다.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 4, 9]

with TemporaryDirectory() as temp_name:
    output_path = Path(temp_name) / "curve.png"

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("x")
    ax.set_ylabel("x squared")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("저장됨:", output_path.exists())
    print("크기 0보다 큼:", output_path.stat().st_size > 0)
```

예상 출력:

```text
저장됨: True
크기 0보다 큼: True
```

여러 그림을 반복 생성할 때 `plt.close(fig)`로 자원을 정리한다. 노트북에서 한글 제목이 네모로 보인다면 데이터 문제가 아니라 시스템 글꼴 설정 문제일 수 있다. 시험에서는 글꼴 설정에 시간을 쓰기보다 영문 축 이름으로 빠르게 EDA하는 것도 방법이다.

## 9. 초심자용 EDA 체크 순서

1. **데이터 구조 확인**: 행 수와 열 이름, 정답 열, ID, 시간·그룹 열을 확인한다.
2. **자료형**: 숫자여야 할 값이 문자열로 읽히지는 않았는지 확인한다.
3. **결측치**: 열별 개수와 비율을 구하고, 특정 그룹에 몰려 있는지 살펴본다.
4. **중복**: 한 관측을 구분하는 기준 열에 중복이 있는지 확인한다.
5. **정답 분포**: 클래스의 종류와 비율을 확인하고, 잘못 입력된 정답 값이 있는지 살펴본다.
6. **수치 데이터의 분포**: 최솟값·최댓값·분위수를 구하고, 히스토그램과 상자그림으로 확인한다.
7. **변수 간 관계**: 정답에 따라 특성의 분포가 다른지 비교하고, 산점도와 상관계수를 확인한다.
8. **데이터 분할**: 같은 차량·장비·사용자의 기록이나 미래 시점의 정보가 학습·검증 데이터에 섞여 누수가 생기지 않는지 확인한다.
9. **확인한 내용 기록**: 그래프마다 무엇을 확인했고 다음에는 무엇을 살펴볼지 한 문장으로 적는다.

그림을 많이 만드는 것이 EDA의 목표가 아니다. 예를 들어 "진동 80은 오류인가 고장 신호인가?", "validation에 처음 보는 장비가 있는가?"처럼 모델링 결정으로 이어지는 질문을 적는다.

## 10. 흔한 오류와 해결법

| 오류·실수 | 원인 | 해결 |
|---|---|---|
| 빈 그래프 또는 축 길이 불일치 | x와 y 행 수가 다름 | plotting 전 `len`, 결측 mask 확인 |
| 그래프가 겹침 | 작은 figure·layout 미조정 | figsize 확대, `tight_layout()` |
| 이전 그림 위에 계속 그려짐 | Figure를 새로 만들지 않음 | `plt.subplots()`와 `plt.close(fig)` |
| histogram 모양만 보고 결론 | bin 설정·표본 수 영향 | describe·여러 bin·도메인 확인 |
| 이상치를 무조건 삭제 | 실제 중요한 고장 신호일 수 있음 | 원인과 평가 목적 확인 |
| 상관을 인과로 해석 | 제3 변수·우연 가능 | 가설로만 사용, 추가 검증 |
| 한글 glyph 경고 | 사용 가능한 한글 글꼴 없음 | 영문 제목 또는 환경별 글꼴 설정 |

## 11. 실습문제

1. class 목록 `['A','A','B','C','A','B']`의 빈도를 막대그래프로 그리고 막대 위에 개수를 표시하라.
2. 수치 배열 `[1,2,2,3,3,3,4,20]`의 histogram과 boxplot을 나란히 그려라. 평균과 중앙값도 출력하라.
3. x `[1,2,3,4,5]`, y `[2,4,5,8,10]` scatter를 그리고 상관계수를 출력하라.
4. DataFrame의 세 수치 열을 반복문을 사용해 1행 3열 histogram으로 그려라.
5. 그룹 A와 B의 value 분포를 boxplot으로 비교하고 그룹별 count·mean·median을 출력하라.
6. 임시 폴더에 그림을 120 dpi PNG로 저장하고 파일이 실제 생성되었는지 확인하라.
7. 작은 DataFrame을 대상으로 EDA 질문 세 개를 말로 작성하라. 최소 하나는 데이터 누수 가능성과 관련되어야 한다.

<details>
<summary>입문 9강 정답·해설 펼치기</summary>

### 1번

```python
import matplotlib.pyplot as plt
import pandas as pd

labels = pd.Series(['A', 'A', 'B', 'C', 'A', 'B'])
counts = labels.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(counts.index, counts.values)
for index, count in enumerate(counts.values):
    ax.text(index, count + 0.05, str(count), ha="center")
ax.set_xlabel("class")
ax.set_ylabel("count")
ax.set_ylim(0, counts.max() + 1)
fig.tight_layout()
plt.show()

print(counts.to_dict())
```

예상 빈도는 A 3, B 2, C 1이다.

### 2번

```python
import matplotlib.pyplot as plt
import numpy as np

values = np.array([1, 2, 2, 3, 3, 3, 4, 20], dtype=float)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].hist(values, bins=6, edgecolor="black")
axes[0].set_title("Histogram")
axes[1].boxplot(values)
axes[1].set_title("Boxplot")
fig.tight_layout()
plt.show()

print("mean:", values.mean())
print("median:", np.median(values))
```

예상 평균은 `4.75`, 중앙값은 `3.0`이다. 20이 평균을 올린다.

### 3번

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 8, 10], dtype=float)

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(x, y)
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.tight_layout()
plt.show()

print(np.corrcoef(x, y)[0, 1])
```

예상 상관계수는 강한 양수이며 약 `0.99`다.

### 4번

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(
    {"x1": [1, 2, 3, 4], "x2": [10, 20, 15, 25], "x3": [0.1, 0.2, 0.2, 0.5]}
)
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, column in zip(axes, df.columns):
    ax.hist(df[column], bins=3, edgecolor="black")
    ax.set_title(column)
fig.tight_layout()
plt.show()
```

x1, x2, x3 열의 히스토그램이 각각 하나씩, 모두 세 개 나타난다.

### 5번

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(
    {"group": ["A", "A", "A", "B", "B", "B"], "value": [1, 2, 3, 5, 7, 9]}
)
a = df.loc[df["group"] == "A", "value"]
b = df.loc[df["group"] == "B", "value"]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.boxplot([a, b])
ax.set_xticks([1, 2])
ax.set_xticklabels(["A", "B"])
ax.set_ylabel("value")
fig.tight_layout()
plt.show()

print(df.groupby("group")["value"].agg(["count", "mean", "median"]))
```

예상 A 평균·중앙값은 2, B는 7이다.

### 6번

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

with TemporaryDirectory() as temp_name:
    path = Path(temp_name) / "practice.png"
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(path.exists())
    print(path.stat().st_size > 0)
```

예상 출력은 `True` 두 줄이다.

### 7번 예시

- 정답 클래스별 개수가 크게 달라 정확도만으로 성능을 판단하기 어려운가?
- vibration의 극단값은 센서 오류인가 실제 fault 신호인가?
- 같은 machine_id의 행이 train과 validation에 동시에 들어가 장비 특성이 누수되지 않는가?

정답 문장은 데이터 문맥에 따라 달라질 수 있다. 중요한 것은 그래프 이름이 아니라 확인할 위험과 다음 행동을 구체적으로 적는 것이다.

</details>

## 완료 기준

- [ ] Figure와 Axes의 관계를 설명하고 `fig, ax = plt.subplots()`로 그린다.
- [ ] 질문에 맞춰 bar·histogram·boxplot·scatter를 선택한다.
- [ ] 모든 그래프에 의미 있는 축 이름을 붙인다.
- [ ] 시각적 패턴을 groupby·describe·상관 같은 숫자로 다시 확인한다.
- [ ] 이상치를 자동으로 삭제하지 않고 원인과 의미를 확인한다.
- [ ] 실습 7문제 중 6문제 이상을 정답 없이 풀었다.

---

# 입문 10강. 미니 프로젝트: CSV를 정리해 PyTorch 텐서 만들기

## 1. 왜 필요한가

지금까지 도구를 따로 배웠다. 이제 CSV 문자열을 읽고, 자료형과 결측을 정리하고, 학습 데이터에서 구한 통계값으로 표준화한 뒤 NumPy 배열과 PyTorch 텐서로 바꾼다. **이번 강의에서는 텐서를 만들고 배치의 모양을 확인하는 것까지 익히면 된다.** 뒤의 신경망 학습은 선택 미리보기이며, 본과정 16~18강을 배운 후 돌아온다.

이 강의의 목적은 높은 점수가 아니다. 단계마다 **행 수, 열 순서, shape, dtype, NaN, 누수 여부**를 확인하는 습관을 만드는 것이다.

이 강의를 마치면 다음을 할 수 있다.

- 원본 표에서 입력 특성과 정답으로 사용할 열을 정한다.
- 학습·검증 데이터를 먼저 나누고, 학습 데이터에서 구한 값으로 결측값을 채우고 표준화한다.
- DataFrame을 float32 NumPy 배열과 Tensor로 변환한다.
- `TensorDataset`, `DataLoader`의 배치 shape를 확인한다.
- 표를 Tensor로 바꾸고 batch의 shape를 확인한다. 모델 학습은 선택 심화다.
- 전체 파이프라인에 assert와 최종 검사를 넣는다.

## 2. 프로젝트 문제와 데이터의 구성

가상의 장비 센서 데이터에서 고장 여부 `target`을 예측한다.

| 항목 | 이 예제의 조건 |
|---|---|
| 한 행 | 한 시점에 측정한 장비의 센서 값 |
| ID | `sample_id`, 모델 입력에서 제외 |
| 입력 특성 | `speed`, `temperature`, `vibration` 세 수치 열 |
| 정답 | 정상 0, 고장 1 |
| split | 앞 12행 train, 뒤 4행 validation인 교육용 고정 split |
| 결측값 | 학습 데이터의 중앙값으로 학습·검증 데이터의 빈칸을 모두 채움 |
| 스케일링 | 학습 데이터의 평균·표준편차로 표준화 |
| 모델 출력 | 샘플당 로짓 하나, 배열 크기는 `[B, 1]` |

실제 시계열 문제에서는 단순 행 위치가 아니라 시간 순서와 장비 그룹을 고려해 split해야 한다. 여기서는 도구 연결에 집중하려고 이미 순서가 정해진 작은 예제를 사용한다.

## 3. 1단계: CSV 읽기와 원본 점검

```python
from io import StringIO
import pandas as pd

csv_text = """sample_id,speed,temperature,vibration,target
S01,40,20,0.10,0
S02,42,21,0.12,0
S03,45,20,0.15,0
S04,47,22,0.18,0
S05,44,,0.14,0
S06,48,23,bad,0
S07,65,30,0.70,1
S08,68,32,0.80,1
S09,70,31,0.75,1
S10,72,34,0.90,1
S11,,33,0.85,1
S12,66,29,0.65,1
S13,43,22,0.11,0
S14,50,24,0.20,0
S15,67,31,0.72,1
S16,74,35,0.95,1
"""

raw = pd.read_csv(StringIO(csv_text), dtype={"sample_id": "string"})

print("shape:", raw.shape)
print("columns:", raw.columns.tolist())
print("dtypes:\n", raw.dtypes)
print("target:\n", raw["target"].value_counts().sort_index())
print("결측:\n", raw.isna().sum())
```

예상 핵심 출력:

```text
shape: (16, 5)
columns: ['sample_id', 'speed', 'temperature', 'vibration', 'target']
target 0: 8개, 1: 8개
speed 결측 1, temperature 결측 1
vibration은 bad 때문에 object/string 계열 dtype
```

`vibration`의 `bad`는 `isna()`에서 아직 결측이 아니다. 문자열을 수치로 변환한 뒤 실패를 다시 점검해야 한다.

## 4. 2단계: 수치 변환과 split

```python
from io import StringIO
import pandas as pd

csv_text = """sample_id,speed,temperature,vibration,target
S01,40,20,0.10,0
S02,42,21,0.12,0
S03,45,20,0.15,0
S04,47,22,0.18,0
S05,44,,0.14,0
S06,48,23,bad,0
S07,65,30,0.70,1
S08,68,32,0.80,1
S09,70,31,0.75,1
S10,72,34,0.90,1
S11,,33,0.85,1
S12,66,29,0.65,1
S13,43,22,0.11,0
S14,50,24,0.20,0
S15,67,31,0.72,1
S16,74,35,0.95,1
"""

feature_columns = ["speed", "temperature", "vibration"]
raw = pd.read_csv(StringIO(csv_text), dtype={"sample_id": "string"})
clean = raw.copy()

for column in feature_columns:
    clean[column] = pd.to_numeric(clean[column], errors="coerce")

train_df = clean.iloc[:12].copy()
valid_df = clean.iloc[12:].copy()

print("변환 후 결측:\n", clean[feature_columns].isna().sum())
print("train/valid:", train_df.shape, valid_df.shape)
print("train target:", train_df["target"].value_counts().sort_index().to_dict())
print("valid target:", valid_df["target"].value_counts().sort_index().to_dict())
```

예상 출력:

```text
speed 1, temperature 1, vibration 1 결측
train/valid: (12, 5) (4, 5)
train target: {0: 6, 1: 6}
valid target: {0: 2, 1: 2}
```

먼저 학습·검증 데이터를 나눈 뒤 학습 데이터의 중앙값·평균·표준편차를 계산해야 한다. 그래야 검증 데이터의 정보가 전처리에 반영되지 않는다.

## 5. 3단계: 학습 데이터의 통계값으로 결측값 처리와 표준화

표준화 공식은 다음과 같다.

```text
표준화 값 = (원래 값 - train 평균) / train 표준편차
```

아래 첫 번째 함수는 학습 데이터에서 구한 통계량을 딕셔너리로 반환한다. 변환 함수는 이 통계량을 새 데이터에 적용한다.

```python
import numpy as np
import pandas as pd

feature_columns = ["speed", "temperature", "vibration"]
train_df = pd.DataFrame(
    {
        "speed": [40.0, 42.0, 45.0, 47.0, 44.0, 48.0, 65.0, 68.0, 70.0, 72.0, np.nan, 66.0],
        "temperature": [20.0, 21.0, 20.0, 22.0, np.nan, 23.0, 30.0, 32.0, 31.0, 34.0, 33.0, 29.0],
        "vibration": [0.10, 0.12, 0.15, 0.18, 0.14, np.nan, 0.70, 0.80, 0.75, 0.90, 0.85, 0.65],
    }
)
valid_df = pd.DataFrame(
    {
        "speed": [43.0, 50.0, 67.0, 74.0],
        "temperature": [22.0, 24.0, 31.0, 35.0],
        "vibration": [0.11, 0.20, 0.72, 0.95],
    }
)


def fit_numeric_preprocessor(frame, columns):
    medians = frame[columns].median()
    filled = frame[columns].fillna(medians)
    means = filled.mean()
    stds = filled.std(ddof=0).replace(0.0, 1.0)
    return {"medians": medians, "means": means, "stds": stds}


def transform_numeric(frame, columns, stats):
    values = frame[columns].copy()
    values = values.fillna(stats["medians"])
    values = (values - stats["means"]) / stats["stds"]
    return values


stats = fit_numeric_preprocessor(train_df, feature_columns)
train_x_df = transform_numeric(train_df, feature_columns, stats)
valid_x_df = transform_numeric(valid_df, feature_columns, stats)

print("train shape:", train_x_df.shape)
print("valid shape:", valid_x_df.shape)
print("NaN 수:", int(train_x_df.isna().sum().sum()), int(valid_x_df.isna().sum().sum()))
print("train 평균:", train_x_df.mean().round(6).to_dict())
print("train 표준편차(ddof=0):", train_x_df.std(ddof=0).round(6).to_dict())
```

예상 출력:

```text
train shape: (12, 3)
valid shape: (4, 3)
NaN 수: 0 0
train 평균은 각 열 약 0
train 표준편차(ddof=0)는 각 열 약 1
```

모든 값이 같은 열은 표준편차가 0이므로, 그대로 표준화하면 0으로 나누게 된다. 예제에서는 `replace(0.0, 1.0)`으로 안전하게 처리했다. 실전에서는 그 열이 의미 있는지 별도로 점검한다.

## 6. 4단계: DataFrame → NumPy → Tensor

```python
import numpy as np
import pandas as pd
import torch

train_x_df = pd.DataFrame(
    {
        "speed": [-1.0, 0.0, 1.0],
        "temperature": [-0.5, 0.0, 0.5],
        "vibration": [-1.2, 0.1, 1.1],
    }
)
train_y_series = pd.Series([0, 0, 1], name="target")

x_np = train_x_df.to_numpy(dtype=np.float32, copy=True)
y_np = train_y_series.to_numpy(dtype=np.int64, copy=True)

x_tensor = torch.from_numpy(x_np)
y_tensor = torch.from_numpy(y_np)

print("NumPy X:", x_np.shape, x_np.dtype)
print("NumPy y:", y_np.shape, y_np.dtype)
print("Tensor X:", tuple(x_tensor.shape), x_tensor.dtype)
print("Tensor y:", tuple(y_tensor.shape), y_tensor.dtype)
print("device:", x_tensor.device)
```

예상 출력:

```text
NumPy X: (3, 3) float32
NumPy y: (3,) int64
Tensor X: (3, 3) torch.float32
Tensor y: (3,) torch.int64
device: cpu
```

`torch.from_numpy`로 만든 CPU Tensor와 NumPy 배열은 메모리를 공유할 수 있다. 여기서는 DataFrame 변환 시 `copy=True`로 원래 분석 표와 메모리를 공유하지 않도록 했다. 이후에도 원본 보존이 중요하면 명시적으로 `.copy()` 또는 Tensor의 `.clone()`을 사용한다.

## 7. 5단계: TensorDataset과 DataLoader

Dataset은 샘플을 보관하고, DataLoader는 여러 샘플을 batch로 묶어 전달한다.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

x = torch.tensor(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    dtype=torch.float32,
)
y = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=3, shuffle=False)

print("샘플 수:", len(dataset))
for batch_index, (batch_x, batch_y) in enumerate(loader):
    print(batch_index, tuple(batch_x.shape), tuple(batch_y.shape))
```

예상 출력:

```text
샘플 수: 4
0 (3, 3) (3,)
1 (1, 3) (1,)
```

마지막 batch는 남은 샘플이 하나라 크기가 1이다. 모델 코드는 마지막 batch도 처리해야 한다. `shuffle=True`는 학습 데이터 순서를 섞을 때 흔히 쓰고 validation/test는 일반적으로 `False`로 둔다.

## 8. 선택 학습: PyTorch 이진분류의 입력과 출력 조건

**여기서 멈추고 실습 1–5, 8번으로 가도 입문 학습은 완성이다.** 아래의 logit·손실함수·역전파·학습 루프는 본과정 16–18강에서 설명한다. 처음 읽는다면 8–9절과 실습 6–7번을 건너뛰자. 8번 실습은 예제의 열 순서만 바꾸면 된다.

`nn.Linear(3, 1)`은 특성 3개를 받아 샘플당 logit 하나를 출력한다. `BCEWithLogitsLoss`는 sigmoid와 binary cross entropy를 안정적으로 합친 손실 함수이므로 모델 안에서 sigmoid를 먼저 적용하지 않는다.

```text
X               [B, 3]  torch.float32
logits          [B, 1]  torch.float32, 제한 없는 실수
target for BCE  [B, 1]  torch.float32, 0 또는 1
probability     sigmoid(logits), 0~1
class           probability >= 0.5
```

```python
import torch
from torch import nn

torch.manual_seed(7)
x = torch.tensor([[0.0, 0.5, -1.0], [1.0, 0.2, 0.7]], dtype=torch.float32)
y = torch.tensor([0, 1], dtype=torch.float32).reshape(-1, 1)

model = nn.Linear(3, 1)
logits = model(x)
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, y)
probabilities = torch.sigmoid(logits)
predictions = (probabilities >= 0.5).to(torch.int64)

print("x:", tuple(x.shape), x.dtype)
print("logits:", tuple(logits.shape), logits.dtype)
print("target:", tuple(y.shape), y.dtype)
print("loss는 유한:", bool(torch.isfinite(loss)))
print("확률 범위 정상:", bool(((probabilities >= 0) & (probabilities <= 1)).all()))
print("예측 shape:", tuple(predictions.shape))
```

예상 출력:

```text
x: (2, 3) torch.float32
logits: (2, 1) torch.float32
target: (2, 1) torch.float32
loss는 유한: True
확률 범위 정상: True
예측 shape: (2, 1)
```

## 9. 선택 심화: 전체 학습 프로젝트 한 셀 실행

이 코드는 앞으로 배울 작업의 전체 모습이다. 지금은 코드를 외우거나 혼자 다시 작성할 필요가 없다. 준비가 되면 셀을 나눠 입력 shape부터 차례로 실행하자.

아래 블록은 데이터 생성부터 validation 평가까지 독립 실행된다. 먼저 실행 결과를 보지 말고 각 `assert`가 무엇을 검사하는지 한 줄씩 읽는다.

```python
from io import StringIO

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 0) 재현성과 데이터 계약
torch.manual_seed(7)
np.random.seed(7)
feature_columns = ["speed", "temperature", "vibration"]

csv_text = """sample_id,speed,temperature,vibration,target
S01,40,20,0.10,0
S02,42,21,0.12,0
S03,45,20,0.15,0
S04,47,22,0.18,0
S05,44,,0.14,0
S06,48,23,bad,0
S07,65,30,0.70,1
S08,68,32,0.80,1
S09,70,31,0.75,1
S10,72,34,0.90,1
S11,,33,0.85,1
S12,66,29,0.65,1
S13,43,22,0.11,0
S14,50,24,0.20,0
S15,67,31,0.72,1
S16,74,35,0.95,1
"""

# 1) 로드와 수치 변환
raw = pd.read_csv(StringIO(csv_text), dtype={"sample_id": "string"})
clean = raw.copy()
for column in feature_columns:
    clean[column] = pd.to_numeric(clean[column], errors="coerce")

assert clean.shape == (16, 5)
assert clean["sample_id"].is_unique
assert set(clean["target"].unique()) == {0, 1}

# 2) split을 통계 계산보다 먼저 수행
train_df = clean.iloc[:12].copy()
valid_df = clean.iloc[12:].copy()
assert set(train_df["sample_id"]).isdisjoint(set(valid_df["sample_id"]))

# 3) train에서만 전처리 통계를 학습
train_medians = train_df[feature_columns].median()
train_filled = train_df[feature_columns].fillna(train_medians)
train_means = train_filled.mean()
train_stds = train_filled.std(ddof=0).replace(0.0, 1.0)


def transform(frame):
    """미리 계산한 train 통계를 사용해 feature를 float32 NumPy 배열로 변환한다."""
    selected = frame[feature_columns].copy()
    selected = selected.fillna(train_medians)
    selected = (selected - train_means) / train_stds
    array = selected.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("전처리 결과에 NaN 또는 inf가 있습니다")
    return array


x_train_np = transform(train_df)
x_valid_np = transform(valid_df)
y_train_np = train_df["target"].to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
y_valid_np = valid_df["target"].to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

assert x_train_np.shape == (12, 3)
assert x_valid_np.shape == (4, 3)
assert y_train_np.shape == (12, 1)
assert y_valid_np.shape == (4, 1)

# 4) NumPy -> CPU Tensor
x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)
x_valid = torch.from_numpy(x_valid_np)
y_valid = torch.from_numpy(y_valid_np)

assert x_train.dtype == torch.float32
assert y_train.dtype == torch.float32

# 5) mini-batch 학습 데이터
train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=4,
    shuffle=True,
    generator=torch.Generator().manual_seed(7),
)

# 6) logit 하나를 내는 작은 MLP
model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

# 7) 학습
model.train()
for epoch in range(120):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()

# 8) validation: gradient 기록 없이 평가
model.eval()
with torch.no_grad():
    valid_logits = model(x_valid)
    valid_probabilities = torch.sigmoid(valid_logits)
    valid_predictions = (valid_probabilities >= 0.5).to(torch.int64)
    valid_targets = y_valid.to(torch.int64)
    valid_accuracy = (valid_predictions == valid_targets).float().mean().item()

# 9) 최종 계약 검사와 결과
assert valid_predictions.shape == (4, 1)
assert torch.isfinite(valid_probabilities).all()
assert ((valid_probabilities >= 0) & (valid_probabilities <= 1)).all()

result = pd.DataFrame(
    {
        "sample_id": valid_df["sample_id"].to_numpy(),
        "probability": valid_probabilities.squeeze(1).numpy(),
        "prediction": valid_predictions.squeeze(1).numpy(),
        "target": valid_targets.squeeze(1).numpy(),
    }
)

print("train X:", tuple(x_train.shape), x_train.dtype)
print("valid X:", tuple(x_valid.shape), x_valid.dtype)
print("validation accuracy:", round(valid_accuracy, 3))
print(result.round({"probability": 4}).to_string(index=False))
```

예상 출력 형식:

```text
train X: (12, 3) torch.float32
valid X: (4, 3) torch.float32
validation accuracy: 1.0
sample_id  probability  prediction  target
      S13        ...             0       0
      S14        ...             0       0
      S15        ...             1       1
      S16        ...             1       1
```

PyTorch·운영체제 버전에 따라 확률의 마지막 소수는 달라질 수 있지만, 데이터가 뚜렷하게 분리되어 일반적인 CPU 실행에서는 네 validation 샘플을 맞히도록 만든 교육용 예제다. 중요한 검사는 숫자 하나가 아니라 shape·dtype·유한값·행 정렬이다.

### 학습 루프를 말로 읽기

1. `model.train()`: 모델을 학습 모드로 둔다.
2. DataLoader가 `(batch_x, batch_y)`를 준다.
3. `optimizer.zero_grad()`: 이전에 계산한 기울기를 지운다.
4. `logits = model(batch_x)`: 순전파한다.
5. `loss = loss_fn(logits, batch_y)`: 오차를 계산한다.
6. `loss.backward()`: 각 매개변수의 기울기를 계산한다.
7. `optimizer.step()`: 매개변수 값을 갱신한다.
8. 평가 때 `model.eval()`과 `torch.no_grad()`를 사용한다.

## 10. 실패를 빠르게 찾는 검사 순서

모델을 바꾸기 전에 다음을 차례로 출력한다.

```python
import numpy as np
import torch

x_np = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
y_np = np.array([[0.0], [1.0]], dtype=np.float32)
x = torch.from_numpy(x_np)
y = torch.from_numpy(y_np)

print("X shape/dtype:", tuple(x.shape), x.dtype)
print("y shape/dtype:", tuple(y.shape), y.dtype)
print("X finite:", bool(torch.isfinite(x).all()))
print("y labels:", torch.unique(y).tolist())
print("same rows:", len(x) == len(y))
```

예상 출력:

```text
X shape/dtype: (2, 2) torch.float32
y shape/dtype: (2, 1) torch.float32
X finite: True
y labels: [0.0, 1.0]
same rows: True
```

실기에서 최소한 확인할 항목:

- 학습 데이터와 테스트 데이터의 입력 특성 열 순서가 같은가?
- ID와 target이 feature에 섞이지 않았는가?
- 전처리 후 NaN·inf가 없는가?
- `X.shape[0] == y.shape[0]`인가?
- 정답 배열의 크기와 자료형이 손실함수의 입력 조건에 맞는가?
- validation에서 `model.eval()`을 호출했는가?
- 예측 행 순서가 원래 제출 ID 순서와 같은가?

## 11. 흔한 오류와 해결법

| 오류·실수 | 원인 | 해결 |
|---|---|---|
| `can't convert np.ndarray of type object` | 숫자 열에 문자열이 남음 | `to_numeric`, dtypes, 실패 수 확인 |
| `mat1 and mat2 shapes cannot be multiplied` | Linear 입력 특성 수 불일치 | `X.shape`, 첫 Linear `in_features` 확인 |
| 손실 계산 시 정답 크기 불일치 | 이진분류 로짓 `[B,1]`, 정답 y `[B]` | y를 `reshape(-1, 1)`로 변경 |
| Float/Double dtype 오류 | NumPy float64가 Tensor float64가 됨 | `to_numpy(dtype=np.float32)` |
| BCE 손실 계산 전에 시그모이드 적용 | `BCEWithLogitsLoss`는 로짓을 입력으로 받음 | 모델은 확률 변환 전의 로짓 출력 |
| 검증 점수가 비현실적으로 높음 | 전체 데이터의 통계값 사용·같은 그룹의 혼합으로 인한 누수 | 데이터를 먼저 나누고 학습 데이터의 통계값만 사용 |
| 매 실행 결과가 크게 달라짐 | seed·shuffle 상태가 다름 | seed와 DataLoader generator 고정 |
| 예측 파일 ID와 예측 순서 불일치 | 정렬·shuffle 후 index 복원 실패 | ID를 따로 보존하고 행 수·순서 assert |

## 12. 실습문제

1. CSV 문자열의 두 수치 열에 `bad`와 빈칸을 넣고 `to_numeric(errors="coerce")` 후 결측 수를 출력하라.
2. 8행 DataFrame을 첫 6행 train, 마지막 2행 validation으로 나눈 뒤 ID가 겹치지 않는지 assert하라.
3. train의 중앙값만 계산해 train과 validation 결측을 채우는 함수를 작성하라.
4. DataFrame을 float32 NumPy 배열로, target을 `(B, 1)` float32 배열로 변환하라.
5. `(5, 3)` 텐서와 `(5, 1)` 정답 텐서로 TensorDataset을 만들고, 배치 크기를 2로 설정했을 때 모든 배치의 배열 크기를 출력하라.
6. **선택 심화:** `nn.Linear(3, 1)`과 `BCEWithLogitsLoss`로 한 번의 optimizer step을 수행하고 loss가 유한한지 확인하라.
7. **선택 심화:** logits `[[-2.0], [0.0], [2.0]]`을 sigmoid 확률과 threshold 0.5 class로 바꾸라.
8. 5절 전처리 예제에서 입력 특성의 열 순서를 `['vibration','speed','temperature']`로 바꾸고 학습·검증 데이터에 동일하게 적용하라. 전체 학습 프로젝트는 실행하지 않아도 된다.

<details>
<summary>입문 10강 정답·해설 펼치기</summary>

### 1번

```python
from io import StringIO
import pandas as pd

csv_text = """x1,x2
1.0,10
bad,20
3.0,
"""
df = pd.read_csv(StringIO(csv_text))
for column in ["x1", "x2"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")
print(df)
print(df.isna().sum())
```

예상 결측 수는 x1 1개, x2 1개다.

### 2번

```python
import pandas as pd

df = pd.DataFrame({"id": [f"S{i}" for i in range(8)], "x": range(8)})
train = df.iloc[:6].copy()
valid = df.iloc[6:].copy()
assert set(train["id"]).isdisjoint(set(valid["id"]))
print(train.shape, valid.shape)
```

예상 출력: `(6, 2) (2, 2)`.

### 3번

```python
import numpy as np
import pandas as pd

train = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [10.0, 20.0, np.nan]})
valid = pd.DataFrame({"x": [np.nan, 100.0], "y": [np.nan, 200.0]})
columns = ["x", "y"]


def fit_medians(frame, selected_columns):
    return frame[selected_columns].median()


def fill_with_medians(frame, selected_columns, medians):
    out = frame.copy()
    out[selected_columns] = out[selected_columns].fillna(medians)
    return out


medians = fit_medians(train, columns)
train_out = fill_with_medians(train, columns, medians)
valid_out = fill_with_medians(valid, columns, medians)
print(medians.to_dict())
print(valid_out)
```

학습 데이터의 중앙값은 x 열 2.0, y 열 15.0이며, 검증 데이터의 결측값을 채울 때도 그대로 사용한다.

### 4번

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"x1": [1, 2, 3], "x2": [10, 20, 30], "target": [0, 1, 0]})
x_np = df[["x1", "x2"]].to_numpy(dtype=np.float32, copy=True)
y_np = df["target"].to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
print(x_np.shape, x_np.dtype)
print(y_np.shape, y_np.dtype)
```

예상 출력: `(3, 2) float32`, `(3, 1) float32`.

### 5번

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

x = torch.arange(15, dtype=torch.float32).reshape(5, 3)
y = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float32).reshape(-1, 1)
loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

for batch_x, batch_y in loader:
    print(tuple(batch_x.shape), tuple(batch_y.shape))
```

예상 batch shape는 `(2,3)/(2,1)`, `(2,3)/(2,1)`, `(1,3)/(1,1)`이다.

### 6번

```python
import torch
from torch import nn

torch.manual_seed(0)
x = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=torch.float32)
y = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

model = nn.Linear(3, 1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

optimizer.zero_grad()
logits = model(x)
loss = loss_fn(logits, y)
loss.backward()
optimizer.step()

print("loss:", float(loss))
print("finite:", bool(torch.isfinite(loss)))
```

예상 마지막 출력: `finite: True`.

### 7번

```python
import torch

logits = torch.tensor([[-2.0], [0.0], [2.0]])
probabilities = torch.sigmoid(logits)
classes = (probabilities >= 0.5).to(torch.int64)
print(probabilities)
print(classes.squeeze(1).tolist())
```

예상 확률은 대략 0.1192, 0.5, 0.8808이고 class는 `[0, 1, 1]`이다. 기준 `>= 0.5`이므로 정확히 0.5도 1이다.

### 8번

```python
import numpy as np
import pandas as pd

df = pd.DataFrame(
    {
        "speed": [40.0, 50.0, 60.0, 70.0],
        "temperature": [20.0, 22.0, 28.0, 32.0],
        "vibration": [0.1, 0.2, 0.6, 0.8],
    }
)
feature_columns = ["vibration", "speed", "temperature"]
train = df.iloc[:3]
valid = df.iloc[3:]

x_train = train[feature_columns].to_numpy(dtype=np.float32, copy=True)
x_valid = valid[feature_columns].to_numpy(dtype=np.float32, copy=True)
print(feature_columns)
print(x_train.shape, x_valid.shape)
print(x_train[0].tolist())
```

예상 첫 행 순서는 `[0.1, 40.0, 20.0]`에 해당한다. 부동소수점 출력에는 아주 작은 표현 차이가 있을 수 있다. 핵심은 두 split 모두 동일한 `feature_columns`를 사용한 것이다.

</details>

## 완료 기준

- [ ] CSV를 읽은 직후 shape·열·dtype·결측·target을 확인한다.
- [ ] split 후 train에서만 결측 대체값과 scaling 통계를 계산한다.
- [ ] 필수 예제의 입력 특성은 `(B, F)` float32, 클래스 라벨은 `(B,)` int64로 만든다. 정답의 크기·자료형 조건은 손실함수마다 다르며 `(B, 1)` float32 변환은 선택 8절에서 다룬다.
- [ ] DataLoader의 마지막 batch 크기가 작아질 수 있음을 안다.
- [ ] Tensor 변환 후 행 수·shape·dtype·NaN을 검사할 수 있다.
- [ ] 필수 실습 1~5, 8번을 풀고, 모르는 부분만 해설로 확인했다.

본과정 16–18강을 배운 뒤에는 선택 심화로 돌아와 로짓·확률·임곗값의 차이, 학습 반복 과정, 예측값 저장을 익히고 실습 6–7번을 푼다. 이 부분을 아직 하지 못해도 입문 과정을 마치는 데는 지장이 없다.

---

# 기초 10강을 마친 뒤: 기존 HDAT-DS 34강으로 넘어가는 법

입문 과정의 목적은 모든 Python 문법을 배우는 것이 아니다. 기존 강의에서 사용되는 데이터를 읽고, shape·dtype 오류를 찾고, 작은 함수를 직접 수정할 기초를 익히는 것이다.

다음 순서로 연결한다.

1. 이 문서의 완료 기준에서 체크하지 못한 항목만 다시 실행한다.
2. 빈 노트북에서 작은 DataFrame을 만들고, 열 두 개를 정해진 순서로 선택해 float32 Tensor로 바꾼다. 10강의 전체 학습 프로젝트를 암기하거나 재작성할 필요는 없다.
3. 기존 HDAT-DS 과정 1강부터 시작한다.
4. 기존 과정에서 NumPy/pandas 코드가 막히면 이 문서 5~8강으로 돌아온다.
5. Tensor shape·dtype가 막히면 6강과 10강에서 입력과 출력 조건을 설명한 표를 다시 본다.
6. 같은 오류로 20분 이상 막힌다면 오류 메시지의 마지막 줄과 입력의 모양·자료형을 확인한다. 그래도 원인을 모르겠다면 오류가 나는 데 꼭 필요한 부분만 남겨 코드를 줄여 본다.

최종 자가 점검:

- [ ] `pd`와 `np`가 무엇의 별명인지 안다.
- [ ] list, ndarray, DataFrame, Tensor를 구분한다.
- [ ] 표를 읽은 뒤 첫 5분에 해야 할 검사를 말할 수 있다.
- [ ] `(B,)`, `(B, 1)`, `(B, F)`를 그림 없이 설명한다.
- [ ] 학습 데이터에서 구한 통계값을 검증·테스트 데이터에 그대로 적용하는 이유를 안다.
- [ ] DataFrame을 float32 Tensor로 바꿀 수 있다.
- [ ] 오류가 나면 메시지를 숨기지 않고 shape·dtype부터 확인한다.

---

# 공식 참고 자료

아래 링크는 본 입문 과정의 개념과 API를 확인할 때 사용한 프로젝트별 **공식 문서**다. 공식 문서는 계속 갱신되므로 시험에서는 해당 회차가 지정한 패키지 버전과 사용 규칙을 우선한다.

## Python 공식 문서

- [Python 자습서: 간단한 소개](https://docs.python.org/3/tutorial/introduction.html)
- [Python 자습서: 자료 구조](https://docs.python.org/3/tutorial/datastructures.html)
- [Python 자습서: 제어 흐름 도구](https://docs.python.org/3/tutorial/controlflow.html)
- [Python 자습서: 오류와 예외](https://docs.python.org/3/tutorial/errors.html)
- [pathlib — 객체 지향 파일 시스템 경로](https://docs.python.org/3/library/pathlib.html)

## NumPy 공식 문서

- [NumPy: absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [NumPy indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html)
- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy copies and views](https://numpy.org/doc/stable/user/basics.copies.html)

## pandas 공식 문서

- [pandas getting started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)
- [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [pandas working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [pandas merge, join, concatenate and compare](https://pandas.pydata.org/docs/user_guide/merging.html)
- [pandas group by](https://pandas.pydata.org/docs/user_guide/groupby.html)

## Matplotlib 공식 문서

- [Matplotlib quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html)
- [Matplotlib pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html)

## PyTorch 공식 문서

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [PyTorch Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [PyTorch Optimization Loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)


---
