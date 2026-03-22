import os
import subprocess

# ===== CONFIGURAÇÕES =====
RAW_FOLDER = "dataset/raw"
PROCESSED_FOLDER = "dataset/processed"
TARGET_WIDTH = 1280
TARGET_FPS = 30

# Cria pasta processed se não existir
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Extensões suportadas
VIDEO_EXTENSIONS = (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI")

def convert_video(input_path, output_path):
    command = [
    "C:/ffmpeg/bin/ffmpeg.exe",
    "-y",
    "-i", input_path,
    "-vf", f"scale={TARGET_WIDTH}:-1,fps={TARGET_FPS}",
    "-an",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    output_path
]


    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main():
    print("🔄 Iniciando pré-processamento automático...\n")

    files = os.listdir(RAW_FOLDER)

    for file in files:
        if file.endswith(VIDEO_EXTENSIONS):

            input_path = os.path.join(RAW_FOLDER, file)

            name, _ = os.path.splitext(file)
            output_name = f"{name}_vo.mp4"
            output_path = os.path.join(PROCESSED_FOLDER, output_name)

            if os.path.exists(output_path):
                print(f"⏭ Já convertido: {file}")
                continue

            print(f"🎬 Convertendo: {file}")
            convert_video(input_path, output_path)
            print(f"✅ Salvo em: {output_name}\n")

    print("🚀 Pré-processamento finalizado!")


if __name__ == "__main__":
    main()
