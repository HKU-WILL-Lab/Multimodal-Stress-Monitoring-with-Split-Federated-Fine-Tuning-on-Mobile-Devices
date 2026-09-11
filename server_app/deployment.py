"""ADB research deployment. Tool paths are local installation settings, never HTTP input."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid


class Deployment:
    def __init__(self, app):
        self.app = app
        self.root = Path(__file__).resolve().parent
        self.settings_file = self.root.parent / 'deployment.json'
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.process = None
        self.state = dict(phase='Idle', message='', logs=[], prepared=False)
        self.settings = json.loads(self.settings_file.read_text(encoding='utf-8')) if self.settings_file.exists() else None
        self.log = self.root / 'runtime' / 'deployment.log'

    def snapshot(self):
        with self.lock:
            result = dict(self.state, available=self.settings is not None)
        if self.log.exists():
            with self.log.open('rb') as f:
                f.seek(max(0, self.log.stat().st_size - 16000))
                result['logs'] = f.read().decode('utf-8', errors='replace').splitlines()[-60:]
        return result

    def call(self, command, timeout=1800):
        if self.cancel.is_set():
            raise RuntimeError('Deployment cancelled')
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
                   PYTHONPATH=str(Path(self.settings['repository']) / 'sfl_runtime/python'))
        env.update(self.settings.get('environment', {}))
        with self.log.open('ab') as log:
            proc = subprocess.Popen([str(a) for a in command], stdout=log, stderr=subprocess.STDOUT,
                env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            self.process = proc
            start = time.monotonic()
            while proc.poll() is None:
                if self.cancel.wait(.2) or time.monotonic()-start > timeout:
                    self.kill()
                    raise RuntimeError('Deployment cancelled or timed out')
            if proc.returncode:
                raise RuntimeError('Command failed; see deployment log: ' + str(command[0]))
            self.process = None

    def kill(self):
        proc = self.process
        if proc and proc.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                proc.terminate()

    def close(self):
        self.cancel.set()
        self.kill()

    def devices(self):
        if not self.settings:
            return []
        result = subprocess.run([self.settings['adb'], 'devices'], capture_output=True, text=True,
            timeout=10, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode:
            raise RuntimeError('ADB could not list devices')
        return [line.split()[0] for line in result.stdout.splitlines()[1:] if len(line.split()) == 2 and line.split()[1] == 'device']

    def begin(self, payload, start=False):
        if not self.settings:
            raise ValueError('Configure the local Android build toolchain first')
        with self.lock:
            if self.state['phase'] == 'Working':
                raise ValueError('A deployment job is already running')
            if start:
                if not self.state.get('prepared'):
                    raise ValueError('Build and deploy the phone configuration first')
                if payload != self.state.get('selection'):
                    raise ValueError('Settings changed. Build and deploy this selection before starting.')
            else:
                if type(payload.get('cut')) is not int or payload['cut'] not in range(1, 5):
                    raise ValueError('Cutting Layer must be 1–4')
                if type(payload.get('steps')) is not int or not 1 <= payload['steps'] <= 100000:
                    raise ValueError('Steps must be 1–100000')
                if payload.get('serial') not in self.devices():
                    raise ValueError('Choose an authorized ADB device')
            self.cancel.clear()
            self.state = dict(phase='Working', message='Starting', prepared=False)
            self.log.parent.mkdir(parents=True, exist_ok=True)
            self.log.write_text('', encoding='utf-8')
        threading.Thread(target=self.work, args=(payload, start), daemon=True).start()

    def message(self, text):
        with self.lock:
            self.state['message'] = text

    def control_phone(self, mode):
        # ADB selects the device explicitly; use an OS-assigned local CDP port.
        adb = self.settings['adb']
        result = subprocess.run([adb, '-s', self.serial, 'shell', 'pidof', 'org.mobihoc.wellbeing'],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        pid = result.stdout.strip()
        if not pid.isdigit():
            raise RuntimeError('Phone application is not running')
        result = subprocess.run([adb, '-s', self.serial, 'forward', 'tcp:0', 'localabstract:webview_devtools_remote_'+pid],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        port = result.stdout.strip()
        if not port.isdigit():
            raise RuntimeError('Cannot open phone debug channel')
        try:
            self.call([self.settings['node'], self.root/'phone_control.cjs', port, mode, self.phone_settings], timeout=60)
        finally:
            subprocess.run([adb, '-s', self.serial, 'forward', '--remove', 'tcp:'+port], capture_output=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def work(self, payload, start):
        try:
            if start:
                self.message('Starting both servers')
                self.app.start('federated')
                self.app.start('main')
                for _ in range(600):
                    if self.cancel.wait(.5):
                        raise RuntimeError('Cancelled')
                    services = self.app.snapshot()['services']
                    if any(s['state'] == 'Failed' for s in services.values()):
                        raise RuntimeError('A server failed to start; see its logs')
                    if all(s['state'] == 'Ready' for s in services.values()):
                        break
                else:
                    raise RuntimeError('Servers did not become ready')
                self.message('Starting the phone training session')
                self.control_phone('start')
                self.state.update(phase='Complete', message='Phone training started', prepared=False)
                return

            self.serial = payload['serial']
            cut, steps = payload['cut'], payload['steps']
            repo = Path(self.settings['repository'])
            adb = [self.settings['adb'], '-s', self.serial]
            self.phone_settings = self.log.parent / 'phone-settings.json'
            # Inspect an existing debug app before replacing it; first installs have no WebView.
            installed = subprocess.run(adb+['shell','pm','path','org.mobihoc.wellbeing'],
                capture_output=True, text=True, timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if 'package:' in installed.stdout:
                self.call(adb + ['shell', 'am', 'start', '-n', 'org.mobihoc.wellbeing/.phone.MainActivity'])
                self.control_phone('probe')
            self.message('Building mobile runtime and APK')
            self.call([self.settings['cmake'], '--build', self.settings['native_build'], '--parallel', '8', '--target', 'wellbeing_sfl'])
            jni = Path(self.settings['jni_dir']) / 'arm64-v8a'
            jni.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(self.settings['native_build'])/'libwellbeing_sfl.so', jni/'libwellbeing_sfl.so')
            gradle = repo/'apps'/('gradlew.bat' if os.name == 'nt' else 'gradlew')
            self.call([gradle, '-p', repo/'apps', ':phone:assembleDebug', '-PenableNativeSfl=true', '-PnativeSflLibDir='+str(jni.parent)])
            self.message('Exporting weights for Cutting Layer '+str(cut))
            model_dir = self.log.parent / ('cut'+str(cut)+'-model')
            self.call([sys.executable.replace('pythonw.exe','python.exe'), '-m', 'sfl_clean.export_llama_embedding',
                '--model-dir', self.app.config.model.source, '--output-dir', model_dir, '--cut-layer', str(cut)])
            self.message('Installing APK and transferring model weights')
            self.call(adb + ['install', '-r', repo/'apps/phone/build/outputs/apk/debug/phone-debug.apk'], timeout=180)
            base = '/sdcard/Android/data/org.mobihoc.wellbeing/files/sfl'
            for required in (self.settings['encoder'], self.settings['dataset']):
                self.call(adb+['shell', 'test', '-f', required], timeout=15)
            mobile_model = base+'/cut'+str(cut)+'-mobile-model'
            self.call(adb+['shell','mkdir','-p',mobile_model])
            for file in model_dir.iterdir():
                self.call(adb+['push',file,mobile_model+'/'+file.name], timeout=600)
            self.message('Creating a new matching server and phone session')
            run = 'd'+str(cut)+'-'+uuid.uuid4().hex[:8]
            directory = self.log.parent/run
            directory.mkdir()
            config = json.loads(json.dumps(self.app.raw))
            config['run_id']=run
            config['training'].update(cut_layer=cut, local_steps=steps)
            config['metrics_path']=str(directory/'server-metrics.jsonl')
            server_file=directory/'server.json'
            server_file.write_text(json.dumps(config,indent=2),encoding='utf-8')
            phone=json.loads(json.dumps(config))
            phone['model']['source']='unused-on-client'
            phone['model']['device']='cpu'
            phone['metrics_path']=base+'/'+run+'/metrics.jsonl'
            phone['checkpoint_root']=base+'/'+run+'/checkpoints'
            phone_file=directory/'client.json'
            phone_file.write_text(json.dumps(phone,indent=2),encoding='utf-8')
            self.call(adb+['shell','mkdir','-p',base+'/'+run])
            self.call(adb+['push',phone_file,base+'/'+run+'/config.json'])
            for remote,host in ((50151,int(config['suffix_rpc']['bind'].rsplit(':',1)[1])),
                                (50052,int(config['coordinator_rpc']['bind'].rsplit(':',1)[1])),(50053,self.app.status_port)):
                self.call(adb+['reverse','tcp:'+str(remote),'tcp:'+str(host)])
            settings=dict(clientId='phone-'+self.serial.replace(':','-'),configPath=base+'/'+run+'/config.json',
                modelDirectory=mobile_model,encoderProgram=self.settings['encoder'],datasetPath=self.settings['dataset'],
                encoderInputLength='240',mainServer='127.0.0.1:50151',federatedServer='127.0.0.1:50052',
                aggregationEndpoint='http://127.0.0.1:50053/status')
            self.phone_settings.write_text(json.dumps(settings),encoding='utf-8')
            self.call(adb+['shell','am','start','-n','org.mobihoc.wellbeing/.phone.MainActivity'])
            self.control_phone('configure')
            if self.cancel.is_set():
                raise RuntimeError('Deployment cancelled')
            self.app.new_config(server_file)
            self.state.update(phase='Complete', message=f'Cutting Layer {cut} deployed; {steps} steps ready',
                              prepared=True, selection=dict(cut=cut, steps=steps, serial=self.serial))
        except Exception as error:
            self.state.update(phase='Failed', message=str(error), prepared=False)
