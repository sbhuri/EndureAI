"""Prompt locally without saving the API key or placing it in shell history."""
import getpass,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if not os.getenv('OPENAI_MODEL'):
    os.environ['OPENAI_MODEL']=input('Responses model available to your account: ').strip()
if not os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY']=getpass.getpass('OpenAI API key (hidden, not saved): ').strip()
from endureai.cli import main
if __name__=='__main__':main()
